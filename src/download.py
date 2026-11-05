"""Fetch Coat and Yahoo! R3 into data/raw. Only downloads what is missing; checks sha256 of downloads."""
import hashlib
import http.client
import io
import urllib.request
import zipfile

from config import RAW

COAT_URL = "https://www.cs.cornell.edu/~schnabts/mnar/coat.zip"
COAT_SHA = "6073d0b515ed1f6e830e4fead66dc76ad7991a7553eaa58e228b234a9d19daed"
YAHOO_URL = "https://raw.githubusercontent.com/DongHande/AutoDebias/main/datasets/yahooR3/"
YAHOO_SHA = {
    "user.txt": "a1fb32e1b74af1227151432db883e425fea8e6c9eb5d31be7aa082af48243275",
    "random.txt": "ef7962a443c31a6edec90e8ae602597acb8a81c680ff33d1c45c5bb64324d0d5",
}


def fetch(url, sha):
    """Connections can drop before the end of the body, so resume with Range until the checksum matches."""
    blob = b""
    for _ in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={len(blob)}-"}), timeout=120) as r:
                if r.status == 200:  # server ignored Range
                    blob = b""
                while chunk := r.read(65536):
                    blob += chunk
        except (OSError, http.client.HTTPException):
            pass
        if hashlib.sha256(blob).hexdigest() == sha:
            print(f"downloaded {url} ({len(blob) / 1e6:.1f} MB)")
            return blob
    raise RuntimeError(f"could not fetch {url} with the expected sha256 ({len(blob)} bytes received)")


def main():
    coat, yahoo = RAW / "coat", RAW / "yahoo"
    coat.mkdir(parents=True, exist_ok=True)
    yahoo.mkdir(parents=True, exist_ok=True)
    if not all((coat / f).exists() for f in ("train.ascii", "test.ascii")):
        with zipfile.ZipFile(io.BytesIO(fetch(COAT_URL, COAT_SHA))) as z:
            for f in ("train.ascii", "test.ascii"):
                (coat / f).write_bytes(z.read(f"coat/{f}"))
    for f, sha in YAHOO_SHA.items():
        if not (yahoo / f).exists():
            (yahoo / f).write_bytes(fetch(YAHOO_URL + f, sha))
    print("data/raw ready")


if __name__ == "__main__":
    main()
