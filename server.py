"""
Road Safety Alert server (v0.2): accounts, search/route proxy, live vehicle sharing.

Run locally:  pip install -r requirements.txt
              uvicorn server:app --port 8000
Environment variables (set them on the host, never in code):
  SECRET_KEY    long random string used to sign login tokens (required in production)
  ORS_API_KEY   optional OpenRouteService key; if absent the free OSRM demo server is used
  DB_PATH       SQLite file path (default app.db)
"""
import asyncio, hashlib, hmac, json, math, os, re, secrets, sqlite3, time
from contextlib import closing

import httpx, jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

SECRET = os.environ.get("SECRET_KEY") or secrets.token_hex(32)   # without SECRET_KEY, logins reset on restart
DB = os.environ.get("DB_PATH", "app.db")
ORS_KEY = os.environ.get("ORS_API_KEY")
UA = {"User-Agent": "road-safety-alert/0.2 (student project)"}     # Nominatim requires an identifying agent

RADIUS_M = 150      # only vehicles within this distance are shared
STALE_S = 10        # forget a vehicle after this many seconds of silence
MAX_ACC_M = 50      # reject GPS fixes worse than this
MIN_GAP_S = 0.4     # ignore updates faster than this
MAX_MSG = 512       # bytes
MAX_VEHICLES = 5000

app = FastAPI()
vehicles = {}       # random anonymous id -> last state (never linked to the account)
hits = {}           # ip -> recent attempt times (login throttling)
geo_cache, geo_lock, last_geo = {}, asyncio.Lock(), 0.0


# ---------- database and passwords ----------
def run(sql, args=()):
    with closing(sqlite3.connect(DB)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, args)
        rows = cur.fetchall()
        c.commit()
        return rows, cur.lastrowid


def init_db():
    run("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, "
        "name TEXT, pw TEXT NOT NULL, created REAL)")


init_db()


def hash_pw(pw, salt=None):
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return salt.hex() + "$" + h.hex()


def check_pw(pw, stored):
    salt, h = stored.split("$")
    return hmac.compare_digest(hash_pw(pw, bytes.fromhex(salt)).split("$")[1], h)


def throttle(request, limit=10, window=300):
    ip = request.client.host if request.client else "?"
    now = time.time()
    recent = [t for t in hits.get(ip, []) if now - t < window]
    if len(recent) >= limit:
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    hits[ip] = recent + [now]


# ---------- tokens ----------
def make_token(uid):
    return jwt.encode({"sub": str(uid), "exp": int(time.time()) + 7 * 86400}, SECRET, algorithm="HS256")


def read_token(t):
    try:
        return int(jwt.decode(t, SECRET, algorithms=["HS256"])["sub"])
    except Exception:
        return None


def auth(authorization: str = Header(default="")):
    uid = read_token(authorization.removeprefix("Bearer ").strip())
    rows, _ = run("SELECT id, email, name FROM users WHERE id=?", (uid,)) if uid else ([], 0)
    if not rows:
        raise HTTPException(401, "Please log in again")
    return dict(rows[0])


# ---------- accounts ----------
class Cred(BaseModel):
    email: str
    password: str
    name: str = ""


@app.post("/api/register")
def register(c: Cred, request: Request):
    throttle(request)
    email, name = c.email.strip().lower(), c.name.strip()[:60]
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Enter a valid email")
    if len(c.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        _, uid = run("INSERT INTO users(email, name, pw, created) VALUES(?,?,?,?)",
                     (email, name, hash_pw(c.password), time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "This email is already registered")
    return {"token": make_token(uid), "user": {"email": email, "name": name}}


@app.post("/api/login")
def login(c: Cred, request: Request):
    throttle(request)
    rows, _ = run("SELECT id, email, name, pw FROM users WHERE email=?", (c.email.strip().lower(),))
    if not rows or not check_pw(c.password, rows[0]["pw"]):
        raise HTTPException(401, "Wrong email or password")
    r = rows[0]
    return {"token": make_token(r["id"]), "user": {"email": r["email"], "name": r["name"]}}


@app.get("/api/me")
def me(user=Depends(auth)):
    return user


# ---------- search and routing (API keys stay on the server) ----------
@app.get("/api/geocode")
async def geocode(q: str, user=Depends(auth)):
    global last_geo
    q = q.strip()[:120]
    if len(q) < 3:
        return []
    if q.lower() in geo_cache:
        return geo_cache[q.lower()]
    async with geo_lock:                        # Nominatim allows at most 1 request per second
        await asyncio.sleep(max(0, 1.1 - (time.time() - last_geo)))
        last_geo = time.time()
        try:
            async with httpx.AsyncClient(timeout=8, headers=UA) as c:
                r = await c.get("https://nominatim.openstreetmap.org/search",
                                params={"q": q, "format": "json", "limit": 5})
            r.raise_for_status()
            out = [{"name": x["display_name"], "lat": float(x["lat"]), "lng": float(x["lon"])} for x in r.json()]
        except (httpx.HTTPError, KeyError, ValueError):
            raise HTTPException(502, "Search service unavailable")
    if len(geo_cache) > 500:
        geo_cache.clear()
    geo_cache[q.lower()] = out
    return out


@app.get("/api/route")
async def route(flat: float, flng: float, tlat: float, tlng: float, user=Depends(auth)):
    try:
        async with httpx.AsyncClient(timeout=20, headers=UA) as c:
            if ORS_KEY:
                r = await c.post("https://api.openrouteservice.org/v2/directions/driving-car/geojson",
                                 headers={"Authorization": ORS_KEY},
                                 json={"coordinates": [[flng, flat], [tlng, tlat]]})
                r.raise_for_status()
                f = r.json()["features"][0]
                pts, dist, dur = f["geometry"]["coordinates"], f["properties"]["summary"]["distance"], f["properties"]["summary"]["duration"]
            else:
                r = await c.get(f"https://router.project-osrm.org/route/v1/driving/{flng},{flat};{tlng},{tlat}",
                                params={"overview": "full", "geometries": "geojson"})
                r.raise_for_status()
                j = r.json()["routes"][0]
                pts, dist, dur = j["geometry"]["coordinates"], j["distance"], j["duration"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        raise HTTPException(502, "Routing service unavailable")
    step = len(pts) // 6000 + 1
    return {"coords": [[y, x] for x, y in pts[::step]] + [[pts[-1][1], pts[-1][0]]], "distance": dist, "duration": dur}


# ---------- live vehicle sharing ----------
def haversine(lat1, lng1, lat2, lng2):
    r, p1, p2 = 6371000, math.radians(lat1), math.radians(lat2)
    dl, dn = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dl / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dn / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def valid(m):
    try:
        lat, lng = float(m["lat"]), float(m["lng"])
        spd, hdg, acc = float(m["spd"]), float(m["hdg"]), float(m["acc"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180 and 0 <= spd <= 80 and 0 <= hdg <= 360):
        return None
    return lat, lng, spd, hdg, acc


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    return {"ok": True, "vehicles": len(vehicles)}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:                                        # first message must be a valid login token
        first = json.loads(await asyncio.wait_for(sock.receive_text(), 10))
        ok = first.get("type") == "auth" and read_token(str(first.get("token", "")))
    except Exception:
        ok = False
    if not ok:
        await sock.close(code=4401)
        return
    if len(vehicles) >= MAX_VEHICLES:
        await sock.close(code=1013)
        return
    vid = secrets.token_hex(4)                  # random per connection, not linked to the account
    await sock.send_text(json.dumps({"type": "hello", "id": vid}))
    try:
        while True:
            raw = await sock.receive_text()
            if len(raw) > MAX_MSG:
                continue
            try:
                data = valid(json.loads(raw))
            except json.JSONDecodeError:
                continue
            if data is None:
                continue
            lat, lng, spd, hdg, acc = data
            now = time.time()
            prev = vehicles.get(vid)
            if prev and now - prev["ts"] < MIN_GAP_S:
                continue
            for k in [k for k, v in vehicles.items() if now - v["ts"] > STALE_S]:
                vehicles.pop(k, None)
            if acc > MAX_ACC_M:                 # poor GPS: do not share, tell the client
                vehicles.pop(vid, None)
                await sock.send_text(json.dumps({"type": "nearby", "others": [], "gps": "poor"}))
                continue
            vehicles[vid] = {"lat": lat, "lng": lng, "spd": spd, "hdg": hdg, "ts": now}
            others = [{"id": oid, "lat": v["lat"], "lng": v["lng"], "spd": v["spd"], "hdg": v["hdg"],
                       "age": round(now - v["ts"], 1)}
                      for oid, v in vehicles.items()
                      if oid != vid and haversine(lat, lng, v["lat"], v["lng"]) <= RADIUS_M]
            await sock.send_text(json.dumps({"type": "nearby", "others": others, "gps": "ok"}))
    except WebSocketDisconnect:
        pass
    finally:
        vehicles.pop(vid, None)
