# MALER legacy production deployment controls

This directory records a secure deployment baseline for the Python 3.7 /
Django 2.1 environment currently used by the Aliyun server.  It does not claim
that an institutional security audit or live TLS inspection has been completed.

## Required deployment sequence

1. Create the Linux environment from `environment_server.yml` in the
   repository root.
2. Store real environment values in a root-owned file with mode `0600`; never
   commit them or paste them into the manuscript.
3. Run `python manage.py audit_security_configuration --strict` in the exact
   service environment.  Deployment must stop on a failed high-risk check.
4. Run `python manage.py check --deploy` and archive the output.
5. On the current Aliyun host, retain the existing Gunicorn/uWSGI and Nginx
   service layout for upload limits, rate limiting, and static assets. A service
   stack migration is not required for this revision. Historical dwebsocket
   calculation routes are disabled; public analyses use the unified fold-local
   validation service.
6. Deny direct HTTP access to `staticfiles/cache`; downloads must pass through
   application views that validate names and paths.
7. Schedule `python manage.py cleanup_maler_cache --apply` at least hourly.
   Confirm the service account can delete only the configured cache subtree.
8. Confirm Nginx and Django use equal or stricter upload limits.
9. Rotate the Django and `.maler` signing secrets independently.
10. Perform the live firewall, access-control, backup, and log audit on the
    actual Aliyun host before describing the deployment as secure. TLS/HTTPS
    configuration is outside the current review deployment scope.

## Honest reporting boundary

The repository provides configuration, automated checks, signed model loading,
path protections, upload caps, and retention tooling.  It cannot establish the
actual firewall, user permissions, backups, or operator practice without
running the checklist on the host. This revision makes no HTTPS claim.
