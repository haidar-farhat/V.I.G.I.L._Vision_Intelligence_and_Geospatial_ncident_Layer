import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Permission } from '@sentinel/shared-types';
import { AuthorizationError, RateLimiter, can, redact } from '@sentinel/security';
import { newRequestId } from '@sentinel/protocol';
import type { Session } from './auth.ts';

/**
 * HTTP routing.
 *
 * Small enough to read in one sitting, which is the point: this is the surface
 * every operator action passes through, and a routing layer nobody fully
 * understands is a routing layer with a hole in it.
 *
 * Two rules are structural rather than conventional:
 *
 * **Every route declares its permission.** There is no "public by default" and no
 * way to register a route without saying who may call it - `permission: null` is
 * an explicit, greppable decision rather than an omission.
 *
 * **Every response carries a request id.** The same id appears in the audit
 * record and in the log line, so a support question about "the export that failed
 * yesterday" is answerable.
 */

export const MAX_BODY_BYTES = 1 * 1024 * 1024;

export type RequestContext = {
  readonly method: string;
  readonly path: string;
  readonly params: Readonly<Record<string, string>>;
  readonly query: URLSearchParams;
  readonly body: unknown;
  readonly session: Session | null;
  readonly requestId: string;
  readonly sourceAddress: string;
};

export type RouteResult = {
  readonly status: number;
  readonly body?: unknown;
  readonly headers?: Readonly<Record<string, string>>;
};

export type Route = {
  readonly method: string;
  /** Path pattern with `:name` segments, e.g. `/api/cameras/:cameraId`. */
  readonly pattern: string;
  /**
   * Permission required. `null` means deliberately unauthenticated - login and
   * health are the only two, and both say so at the call site.
   */
  readonly permission: Permission | null;
  readonly handler: (context: RequestContext) => Promise<RouteResult> | RouteResult;
  /** Per-route rate limit, for anything expensive or brute-forceable. */
  readonly limit?: { readonly attempts: number; readonly windowMillis: number };
};

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly recoverable: boolean;

  constructor(status: number, code: string, message: string, recoverable = true) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.recoverable = recoverable;
  }
}

type CompiledRoute = Route & {
  readonly segments: readonly string[];
  readonly limiter: RateLimiter | null;
};

const compile = (route: Route): CompiledRoute => ({
  ...route,
  segments: route.pattern.split('/').filter((segment) => segment !== ''),
  limiter:
    route.limit === undefined
      ? null
      : new RateLimiter(route.limit.attempts, route.limit.windowMillis),
});

/** Match a path against a pattern, extracting `:name` parameters. */
const match = (
  route: CompiledRoute,
  segments: readonly string[],
): Record<string, string> | null => {
  if (route.segments.length !== segments.length) return null;

  const params: Record<string, string> = {};
  for (let i = 0; i < route.segments.length; i += 1) {
    const expected = route.segments[i] ?? '';
    const actual = segments[i] ?? '';

    if (expected.startsWith(':')) {
      // A path parameter is decoded once, here. Decoding it again downstream is
      // how a %252e%252e becomes a traversal after two passes.
      params[expected.slice(1)] = decodeURIComponent(actual);
      continue;
    }
    if (expected !== actual) return null;
  }
  return params;
};

export type RouterOptions = {
  /** Resolve a session from a bearer token. */
  readonly resolveSession: (token: string) => Session | undefined;
  /** Called for every request once it has been authorised or refused. */
  readonly onAudit?: (entry: AuditEntry) => void;
  readonly onLog?: (entry: LogEntry) => void;
};

export type AuditEntry = {
  readonly requestId: string;
  readonly method: string;
  readonly path: string;
  readonly username: string | null;
  readonly permission: Permission | null;
  readonly outcome: 'SUCCESS' | 'DENIED' | 'ERROR';
  readonly status: number;
  readonly sourceAddress: string;
};

export type LogEntry = {
  readonly requestId: string;
  readonly level: 'info' | 'warn' | 'error';
  readonly message: string;
  readonly detail?: unknown;
};

export class Router {
  readonly #routes: CompiledRoute[] = [];
  readonly #options: RouterOptions;

  constructor(options: RouterOptions) {
    this.#options = options;
  }

  add(route: Route): this {
    this.#routes.push(compile(route));
    return this;
  }

  get routeCount(): number {
    return this.#routes.length;
  }

  /**
   * Handle one request.
   *
   * The order is deliberate: match, then authenticate, then rate-limit, then
   * authorise, then run. Rate limiting an unauthenticated caller before checking
   * their permission means a flood cannot be used to probe which routes exist.
   */
  async handle(
    method: string,
    url: string,
    headers: Readonly<Record<string, string | string[] | undefined>>,
    body: unknown,
    sourceAddress: string,
    now: number = Date.now(),
  ): Promise<RouteResult & { readonly requestId: string }> {
    const requestId = String(newRequestId());

    let parsed: URL;
    try {
      parsed = new URL(url, 'http://localhost');
    } catch {
      return { ...error(400, 'BAD_URL', 'The request URL could not be parsed.'), requestId };
    }

    const segments = parsed.pathname.split('/').filter((segment) => segment !== '');

    let route: CompiledRoute | undefined;
    let params: Record<string, string> = {};
    let pathMatchedOtherMethod = false;

    for (const candidate of this.#routes) {
      const matched = match(candidate, segments);
      if (matched === null) continue;

      if (candidate.method === method) {
        route = candidate;
        params = matched;
        break;
      }
      pathMatchedOtherMethod = true;
    }

    if (route === undefined) {
      // 405 rather than 404 when the path exists under another verb: a client
      // that used the wrong method deserves to be told so.
      return pathMatchedOtherMethod
        ? { ...error(405, 'METHOD_NOT_ALLOWED', `${method} is not allowed on this path.`), requestId }
        : { ...error(404, 'NOT_FOUND', 'No such endpoint.'), requestId };
    }

    // --- authenticate -------------------------------------------------------
    const authorization = headerValue(headers, 'authorization');
    const token = authorization.startsWith('Bearer ') ? authorization.slice(7).trim() : '';
    const session = token === '' ? null : (this.#options.resolveSession(token) ?? null);

    // --- rate limit ---------------------------------------------------------
    if (route.limiter !== null) {
      const key = session === null ? `addr:${sourceAddress}` : `user:${session.userId}`;
      const attempt = route.limiter.attempt(`${route.pattern}:${key}`, now);

      if (!attempt.allowed) {
        this.#audit({
          requestId,
          method,
          path: parsed.pathname,
          username: session?.username ?? null,
          permission: route.permission,
          outcome: 'DENIED',
          status: 429,
          sourceAddress,
        });
        return {
          ...error(429, 'RATE_LIMITED', 'Too many requests. Try again shortly.'),
          headers: { 'Retry-After': String(Math.ceil((attempt.resetAt - now) / 1000)) },
          requestId,
        };
      }
    }

    // --- authorise ----------------------------------------------------------
    if (route.permission !== null) {
      if (session === null) {
        this.#audit({
          requestId,
          method,
          path: parsed.pathname,
          username: null,
          permission: route.permission,
          outcome: 'DENIED',
          status: 401,
          sourceAddress,
        });
        return {
          ...error(401, 'UNAUTHENTICATED', 'This endpoint requires an authenticated session.'),
          requestId,
        };
      }

      if (!can({ roles: session.roles, active: true }, route.permission)) {
        this.#audit({
          requestId,
          method,
          path: parsed.pathname,
          username: session.username,
          permission: route.permission,
          outcome: 'DENIED',
          status: 403,
          sourceAddress,
        });
        // Names what was required, never what the caller holds - enumerating a
        // user's permissions to an unauthorised caller is itself a leak.
        return {
          ...error(403, 'FORBIDDEN', new AuthorizationError(route.permission).message),
          requestId,
        };
      }
    }

    // --- run ----------------------------------------------------------------
    try {
      const result = await route.handler({
        method,
        path: parsed.pathname,
        params,
        query: parsed.searchParams,
        body,
        session,
        requestId,
        sourceAddress,
      });

      this.#audit({
        requestId,
        method,
        path: parsed.pathname,
        username: session?.username ?? null,
        permission: route.permission,
        outcome: result.status < 400 ? 'SUCCESS' : 'DENIED',
        status: result.status,
        sourceAddress,
      });

      return { ...result, requestId };
    } catch (caught) {
      const apiError =
        caught instanceof ApiError
          ? caught
          : new ApiError(500, 'INTERNAL', 'The request could not be completed.', true);

      // The real error is logged, redacted; the caller gets the sanitised one.
      // An internal message can carry a file path, a query or a credential, and
      // none of those belong in a response.
      this.#options.onLog?.({
        requestId,
        level: 'error',
        message: caught instanceof Error ? caught.message : String(caught),
        detail: redact({ path: parsed.pathname, method }),
      });

      this.#audit({
        requestId,
        method,
        path: parsed.pathname,
        username: session?.username ?? null,
        permission: route.permission,
        outcome: 'ERROR',
        status: apiError.status,
        sourceAddress,
      });

      return {
        ...error(apiError.status, apiError.code, apiError.message, apiError.recoverable),
        requestId,
      };
    }
  }

  #audit(entry: AuditEntry): void {
    this.#options.onAudit?.(entry);
  }
}

const error = (
  status: number,
  code: string,
  message: string,
  recoverable = true,
): RouteResult => ({
  status,
  body: { error: { code, message, recoverable } },
});

export const headerValue = (
  headers: Readonly<Record<string, string | string[] | undefined>>,
  name: string,
): string => {
  const value = headers[name] ?? headers[name.toLowerCase()];
  if (value === undefined) return '';
  return Array.isArray(value) ? (value[0] ?? '') : value;
};

/**
 * Read and parse a JSON request body.
 *
 * Capped, and the cap is enforced as bytes arrive rather than after. Waiting for
 * the whole body before checking its size is how a single request consumes a
 * gigabyte of heap.
 */
export const readJsonBody = async (request: IncomingMessage): Promise<unknown> =>
  new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;

    request.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        request.destroy();
        reject(
          new ApiError(413, 'BODY_TOO_LARGE', `Request body exceeds ${MAX_BODY_BYTES} bytes.`, false),
        );
        return;
      }
      chunks.push(chunk);
    });

    request.on('end', () => {
      if (chunks.length === 0) {
        resolve(undefined);
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(new ApiError(400, 'BAD_JSON', 'Request body is not valid JSON.', false));
      }
    });

    request.on('error', (cause) => reject(cause));
  });

/** Write a route result, with the headers every response carries. */
export const writeResult = (
  response: ServerResponse,
  result: RouteResult & { readonly requestId: string },
): void => {
  const body = result.body === undefined ? '' : JSON.stringify(result.body);

  response.writeHead(result.status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body, 'utf8'),
    'X-Request-Id': result.requestId,
    // This API serves a local desktop app over loopback and is never embedded.
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Cache-Control': 'no-store',
    ...result.headers,
  });

  response.end(body);
};
