# Security and data protection

## What is protected and how
| Concern | Measure |
| --- | --- |
| Passwords | bcrypt hashing (cost 12); never stored or logged in clear text; minimum 10 characters with letters and digits |
| Sessions | signed JWT bearer tokens (HS256) valid 8 h; secret generated per installation in `data/.jwt_secret` (or `*_JWT_SECRET`) |
| Brute force | per-IP rate limit (10 attempts / 5 min) **and** per-account lockout (5 failures → 15 min) |
| Roles | administrator rights are bound to the configured e-mail list (`*_ADMIN_EMAILS`, default the platform owner); every other account is a researcher |
| Data isolation | researchers can read, download, annotate or delete **only their own** objects, images, faces and analyses; the administrator sees everything (users, audit log, all analyses, flags, corpus) |
| Uploads | type and size checks, immutable originals, SHA-256 of every file, path-safe artefact access (no path traversal) |
| Browser | Content-Security-Policy, no-store caching, X-Frame-Options DENY, nosniff, same-origin referrer |
| Network | the server binds to 127.0.0.1 only — it is not reachable from other computers unless you deploy it behind HTTPS on purpose |
| Audit | every registration, sign-in (and failure), upload, download, annotation and administrative action is logged with time, user and IP |
| Museum data | licence and cultural-sensitivity fields decide whether an image may be used for training; restricted images are stored as metadata only |

## Where the data is
Everything is on this computer, inside the application folder: `data/platform.db` (SQLite: users, objects, images, faces, annotations, audit log), `data/uploads/` (originals), `data/analyses/` (results and figures), `data/corpus/` or `data/mochefacedb/` (reference images), `data/jobs/` (background-job logs). No data leaves the computer except the requests the administrator triggers to public museum APIs when building the reference corpus.

## Recommendations for the owner
1. Keep the application folder on an encrypted drive (Windows BitLocker) — the platform does not encrypt the database itself.
2. Back up `data/` regularly; the `.jwt_secret` file must stay private (delete it to invalidate all sessions).
3. To open the platform to other computers use a reverse proxy with HTTPS and set `*_CORS_ORIGINS`; never expose port 8765 directly.
4. Set `*_ADMIN_EMAILS` to change the administrator list; existing accounts are re-evaluated at their next sign-in.


