"""The operator console: a thin view over the service.

It holds no domain state. Every change to the site goes through `Commands`,
which carries the principal and turns a refusal into a sentence. The window
draws what `Runtime.poll()` returns and what `SiteService` reports, and
nothing else.
"""
