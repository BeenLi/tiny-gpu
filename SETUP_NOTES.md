# tiny-gpu setup on myDevbox

Host: myDevbox (Debian 10 buster, glibc 2.28). GitHub reachable only via the
local proxy http://127.0.0.1:5555 (already in ~/.gitconfig for github.com).

## How to run
```sh
cd /data00/home/wanli.99/autoResearch/tiny-gpu
source env.sh            # activates .venv (cocotb 1.9.2) + puts ~/.local/bin on PATH
make test_matadd         # PASS
make test_matmul         # PASS
```

## Toolchain installed
- iverilog / vvp 10.2        -> apt (system /usr/bin)
- cocotb 1.9.2               -> ~/autoResearch/tiny-gpu/.venv (Python 3.11 via uv)
                               NOTE: global ~/.local cocotb is 2.0 on py3.7 and is
                               NOT compatible (cocotb 2.0 renamed the MODULE env var).
- sv2v v0.0.13               -> ~/.local/libexec/sv2v.real + wrapper ~/.local/bin/sv2v

## Why the sv2v wrapper exists
The official sv2v Linux binary needs GLIBC_2.34, but buster has 2.28. The wrapper
~/.local/bin/sv2v runs the real binary under a newer glibc (2.36) extracted from a
Debian bookworm libc6 .deb into ~/.local/glibc236, via:
  ld-linux-x86-64.so.2 --library-path <glibc236>:<system libs> sv2v.real "$@"
(system libgmp.so.10 is still picked up from the system path.)
