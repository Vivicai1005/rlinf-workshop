"""Build-time downloads with retries (connections from the build cluster to github.com drop intermittently).

    python3 fetch.py src  <owner/repo> <40-hex commit> <dest dir>     # GitHub source tarball via codeload
    python3 fetch.py file <sha256> <dest file> <url> [<url> ...]      # first URL that works, checked against sha256

Files are written to disk (the RoboTwin assets are several GB each) and verified before they are kept. Files of
64 MB or more are fetched as 32 MB byte ranges over 8 parallel connections when the server supports ranges (Hugging
Face's CDN limits each connection to ~5 MB/s; 8 connections measured ~39 MB/s), each range with its own retries;
otherwise as one stream. A stalled connection fails after 2 minutes, a trickling range after 10; with several URLs each gets fewer attempts, so a
dead one does not use up the build.
"""
import concurrent.futures, hashlib, os, re, shutil, sys, tarfile, time, urllib.request

UA = {"User-Agent": "rlinf-workshop-image-build/1.0"}  # some CDNs (releases.astral.sh) refuse Python's default one
TIMEOUT = 120
PARALLEL_MIN, PART, CONNECTIONS = 64 * 2**20, 32 * 2**20, 8
PART_DEADLINE = 600  # a range still unfinished after 10 minutes is being trickled (CDN throttling): retry it


def download(url, out, attempts=6):
    """One stream, restarted from the beginning on failure."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=TIMEOUT) as r, \
                    open(out, "wb") as f:
                shutil.copyfileobj(r, f, 8 * 2**20)
            return True
        except Exception as e:  # noqa: BLE001 -- retry any network error
            print(f"attempt {attempt} for {url} failed: {e}", flush=True)
            time.sleep(20)
    return False


def ranged_size(url):
    """Total size if the server (after redirects) answers byte ranges, else None."""
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            m = re.match(r"bytes 0-0/(\d+)$", r.headers.get("Content-Range", ""))
            return int(m.group(1)) if r.status == 206 and m else None
    except Exception:  # noqa: BLE001 -- no ranges: fall back to one stream
        return None


def download_ranges(url, out, size, attempts=4):
    """32 MB ranges over parallel connections, written in place; each range is retried on its own."""
    with open(out, "wb") as f:
        f.truncate(size)
    fd = os.open(out, os.O_WRONLY)

    def part(start):
        end = min(size, start + PART) - 1
        for attempt in range(1, attempts + 1):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    assert r.status == 206, f"HTTP {r.status} for a range request"
                    pos, t_part = start, time.time()
                    while pos <= end:
                        if time.time() - t_part > PART_DEADLINE:
                            raise TimeoutError(f"only {pos - start} bytes in {PART_DEADLINE} s")
                        block = r.read(min(2**20, end - pos + 1))
                        if not block:
                            raise IOError(f"range {start}-{end} ended at {pos}")
                        os.pwrite(fd, block, pos)
                        pos += len(block)
                return end - start + 1
            except Exception as e:  # noqa: BLE001 -- retry this range
                print(f"range {start}-{end} attempt {attempt} failed: {e}", flush=True)
                time.sleep(10)
        raise IOError(f"range {start}-{end} failed {attempts} times")

    t0, done = time.time(), 0
    try:
        with concurrent.futures.ThreadPoolExecutor(CONNECTIONS) as pool:
            for n in pool.map(part, range(0, size, PART)):
                done += n
                if done % (16 * PART) < PART or done == size:
                    print(f"  {done / 2**30:.1f}/{size / 2**30:.1f} GB, {done / 2**20 / (time.time() - t0):.0f} MB/s",
                          flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"ranged download of {url} failed: {e}", flush=True)
        return False
    finally:
        os.close(fd)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


kind = sys.argv[1] if len(sys.argv) > 1 else ""
if kind == "src":
    repo, commit, dest = sys.argv[2:5]
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "pin a full commit"
    tmp = dest.rstrip("/") + ".tar.gz"
    if not download(f"https://codeload.github.com/{repo}/tar.gz/{commit}", tmp):
        sys.exit(f"could not download {repo}@{commit}")
    parent = os.path.dirname(dest.rstrip("/")) or "."
    os.makedirs(parent, exist_ok=True)
    with tarfile.open(tmp) as t:
        root = t.getnames()[0].split("/")[0]
        try:
            # "tar", not "data": repositories contain relative symlinks (RLinf's .claude/skills) that git checks out
            # as they are; absolute paths and members outside the destination are still refused.
            t.extractall(parent, filter="tar")
        except TypeError:  # Python without extraction filters
            t.extractall(parent)
    if os.path.isdir(dest) and not os.listdir(dest):  # an empty git submodule directory
        os.rmdir(dest)
    os.rename(os.path.join(parent, root), dest)
    os.remove(tmp)
    print(f"{repo}@{commit[:12]} -> {dest}")
elif kind == "file":
    digest, out, urls = sys.argv[2], sys.argv[3], sys.argv[4:]
    for url in urls:
        attempts = max(2, 6 // len(urls))
        size = ranged_size(url)
        if size and size >= PARALLEL_MIN:
            ok = download_ranges(url, out, size) or download(url, out, attempts)
        else:
            ok = download(url, out, attempts)
        if ok and sha256(out) == digest:
            print(f"{os.path.basename(out)}: {os.path.getsize(out)} bytes, sha256 ok ({url.split('/')[2]})")
            break
        print(f"{url}: download failed or sha256 mismatch", flush=True)
    else:
        sys.exit(f"could not get {os.path.basename(out)} with sha256 {digest}")
else:
    sys.exit(__doc__)
