# MALER R2 public research deployment

Python 3.7.11/Django 2.1.8 are retained and unsupported. These operational
controls do not provide runtime security support or clinical certification.

## Scope

The public HTTPS service accepts authorized, non-sensitive research data.
Analysis and Predict require the displayed data-policy acknowledgment. Sensitive
or confidential data need an appropriately controlled deployment.

## Private volatile storage

The public task root is /var/lib/maler_public_r2/tasks on a 1.5-GiB tmpfs.
Dedicated private directories hold Django upload buffers and temporary work;
Nginx upload buffering uses a separate nginx-owned directory on the same mount.
The mount is nodev,nosuid,noexec, configured persistently by a systemd mount
unit. No swap is configured on the checked host. Storage initialization and
application units order startup after the mount. New public task files are not
copied by MALER into NAS task roots. This does not certify absence of provider,
legacy, audit or other copies. Reboot/failure can remove active data early.

Access tokens expire after 24 hours. A five-minute systemd timer runs
cleanup_maler_cache --apply in the actual public environment. Verify an expired
synthetic fixture is removed and an unexpired fixture remains during a real
scheduled run. No all-copy 24-hour erasure guarantee is made.

## Runtime and request controls

- Preserve the root-owned 0600 environment file and independent random Django
  and model-signing keys. Do not expose keys in source, responses or evidence.
- Keep DEBUG=False, allowed hosts explicit, secure cookies and HTTPS redirects.
- Bind Gunicorn to loopback, with two workers and a 900-second worker timeout.
- Apply a 4-GiB service memory ceiling, two-CPU quota, task limit and no core dumps.
- Limit data/model uploads to 50/100 MiB, multipart requests to 160 MiB.
- Keep at least 128 MiB plus declared incoming body size free in task storage;
  reject admission before multipart parsing if capacity is insufficient.
- Limit Nginx request/upload rates, per-client connections and simultaneous
  compute requests. Keep access logs free of query-string capability tokens.
- Disable direct static-cache access and historical WebSocket calculations.
- Keep external framework/runtime migration as outstanding maintenance work.

## Release and verification

Run the exact-source 95-test suite before switching the service. Run strict
configuration audit and Django deployment checks in the actual environment,
and record any warnings. Verify actual process/source/interpreter identity.
Use the real certificate-verifying public HTTPS entry for a synthetic browser
upload -> preview -> development -> result -> signed export -> Predict sequence.
Independently compare displayed predictions with the server's original signed
pipeline, lock and model hashes without fitting. Verify automatic expiry and
explicit deletion, clean up only identified synthetic tasks and keep the public
service available. Preserve rollback configuration privately.

Historical deployment instructions are in docs/history/deployment_status_20260930.md.
