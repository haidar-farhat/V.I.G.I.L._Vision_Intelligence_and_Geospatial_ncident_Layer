/**
 * Zero-WAN enforcement.
 *
 * The platform's headline guarantee is that it never needs the Internet. That
 * guarantee is only worth anything if it is enforced by code rather than by
 * intention: a dependency added in six months, a well-meaning "check for
 * updates", or a map style with one absolute URL in it would quietly break it and
 * nobody would notice until an auditor asked.
 *
 * So every outbound address is classified before a connection is attempted, and
 * anything outside the private ranges is refused with a specific, explanatory
 * error. The failure is loud by design - silently falling back to an online
 * service is the one behaviour this system must never exhibit.
 */

export const AddressScope = {
  /** Loopback: 127.0.0.0/8, ::1. */
  Loopback: 'LOOPBACK',
  /** RFC1918, RFC4193, and carrier-grade NAT space. */
  Private: 'PRIVATE',
  /** Link-local: 169.254.0.0/16, fe80::/10. */
  LinkLocal: 'LINK_LOCAL',
  /** Multicast, including the mDNS group. */
  Multicast: 'MULTICAST',
  /** Anything routable on the public Internet. */
  Public: 'PUBLIC',
  /** A hostname that is not an IP literal, so scope cannot be decided here. */
  Unresolved: 'UNRESOLVED',
} as const;
export type AddressScope = (typeof AddressScope)[keyof typeof AddressScope];

/** Hostnames that always resolve locally and need no DNS. */
const LOCAL_HOSTNAMES: readonly string[] = Object.freeze([
  'localhost',
  'localhost.localdomain',
  '::1',
]);

/** Suffixes reserved for local-link name resolution (mDNS and friends). */
const LOCAL_SUFFIXES: readonly string[] = Object.freeze(['.local', '.localhost', '.home.arpa']);

const parseIpv4 = (host: string): readonly number[] | null => {
  const parts = host.split('.');
  if (parts.length !== 4) return null;

  const octets: number[] = [];
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) return null;
    const value = Number(part);
    if (value > 255) return null;
    octets.push(value);
  }
  return octets;
};

const classifyIpv4 = (octets: readonly number[]): AddressScope => {
  const a = octets[0] ?? 0;
  const b = octets[1] ?? 0;

  if (a === 127) return AddressScope.Loopback;
  if (a === 10) return AddressScope.Private;
  if (a === 172 && b >= 16 && b <= 31) return AddressScope.Private;
  if (a === 192 && b === 168) return AddressScope.Private;
  // Carrier-grade NAT; treated as private because some appliances sit there.
  if (a === 100 && b >= 64 && b <= 127) return AddressScope.Private;
  if (a === 169 && b === 254) return AddressScope.LinkLocal;
  if (a >= 224 && a <= 239) return AddressScope.Multicast;
  if (a === 0) return AddressScope.LinkLocal;

  return AddressScope.Public;
};

const classifyIpv6 = (host: string): AddressScope | null => {
  const normalised = host.toLowerCase().replace(/^\[|\]$/g, '');
  if (!normalised.includes(':')) return null;

  if (normalised === '::1') return AddressScope.Loopback;
  if (normalised === '::') return AddressScope.LinkLocal;

  // Unique local addresses, fc00::/7.
  if (/^f[cd][0-9a-f]{2}:/.test(normalised)) return AddressScope.Private;
  // Link-local, fe80::/10.
  if (/^fe[89ab][0-9a-f]:/.test(normalised)) return AddressScope.LinkLocal;
  // Multicast, ff00::/8.
  if (/^ff[0-9a-f]{2}:/.test(normalised)) return AddressScope.Multicast;
  // IPv4-mapped, ::ffff:a.b.c.d - classify by the embedded IPv4 address.
  const mapped = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/.exec(normalised);
  if (mapped?.[1] !== undefined) {
    const octets = parseIpv4(mapped[1]);
    if (octets !== null) return classifyIpv4(octets);
  }

  return AddressScope.Public;
};

/** Classify a host, which may be an IP literal or a hostname. */
export const classifyHost = (host: string): AddressScope => {
  const trimmed = host.trim().toLowerCase();
  if (trimmed === '') return AddressScope.Unresolved;

  if (LOCAL_HOSTNAMES.includes(trimmed)) return AddressScope.Loopback;

  const ipv6 = classifyIpv6(trimmed);
  if (ipv6 !== null) return ipv6;

  const ipv4 = parseIpv4(trimmed);
  if (ipv4 !== null) return classifyIpv4(ipv4);

  if (LOCAL_SUFFIXES.some((suffix) => trimmed.endsWith(suffix))) return AddressScope.Private;

  // A public DNS name would require an Internet resolver to reach, so for the
  // purposes of this guard it is not local. It is reported as Unresolved rather
  // than Public so the caller can distinguish "definitely public" from "cannot
  // prove local", and refuse both without conflating them.
  return AddressScope.Unresolved;
};

/** Scopes reachable without the Internet. */
export const LOCAL_SCOPES: readonly AddressScope[] = Object.freeze([
  AddressScope.Loopback,
  AddressScope.Private,
  AddressScope.LinkLocal,
  AddressScope.Multicast,
]);

export const isLocalScope = (scope: AddressScope): boolean => LOCAL_SCOPES.includes(scope);

export class EgressBlockedError extends Error {
  readonly code = 'EGRESS_BLOCKED';
  readonly host: string;
  readonly scope: AddressScope;
  readonly recoverable = false;

  constructor(host: string, scope: AddressScope) {
    super(
      `Refused to contact "${host}" (${scope}). Sentinel Vision operates without ` +
        'Internet access by design and never falls back to an online service. ' +
        'If this address is on your LAN, configure it by IP or a .local name.',
    );
    this.name = 'EgressBlockedError';
    this.host = host;
    this.scope = scope;
  }
}

export type EgressGuardOptions = {
  /**
   * When false the guard reports but does not block, for development against
   * external test fixtures. Never false in a packaged build.
   */
  readonly enforce?: boolean;
  /** Additional hostnames explicitly permitted by the operator. */
  readonly allowHosts?: readonly string[];
};

/**
 * The egress guard.
 *
 * Every outbound connection - camera, node, LLM endpoint, map source - passes
 * through `check` before a socket is opened.
 */
export class EgressGuard {
  readonly #enforce: boolean;
  readonly #allow: ReadonlySet<string>;
  #blockedCount = 0;
  #allowedCount = 0;

  constructor(options: EgressGuardOptions = {}) {
    this.#enforce = options.enforce ?? true;
    this.#allow = new Set((options.allowHosts ?? []).map((h) => h.trim().toLowerCase()));
  }

  get enforcing(): boolean {
    return this.#enforce;
  }

  get blockedCount(): number {
    return this.#blockedCount;
  }

  get allowedCount(): number {
    return this.#allowedCount;
  }

  /** Whether a host may be contacted, without throwing. */
  permits(host: string): boolean {
    if (this.#allow.has(host.trim().toLowerCase())) return true;
    return isLocalScope(classifyHost(host));
  }

  /**
   * Assert that a host may be contacted. Throws `EgressBlockedError` otherwise.
   */
  check(host: string): void {
    if (this.permits(host)) {
      this.#allowedCount += 1;
      return;
    }

    this.#blockedCount += 1;
    if (this.#enforce) throw new EgressBlockedError(host, classifyHost(host));
  }

  /** Extract the host from a URL and check it. */
  checkUrl(url: string): void {
    let host: string;
    try {
      host = new URL(url).hostname;
    } catch {
      throw new EgressBlockedError(url, AddressScope.Unresolved);
    }
    this.check(host);
  }
}

/**
 * The network isolation report backing the diagnostic screen.
 *
 * Stated plainly so an operator - or an auditor - can read the guarantee off one
 * panel rather than inferring it from configuration.
 */
export type NetworkIsolationReport = {
  readonly wan: 'BLOCKED' | 'NOT_REQUIRED';
  readonly lan: 'ACTIVE' | 'INACTIVE';
  readonly internetDependency: 'NONE';
  readonly cloudServices: 'DISABLED';
  readonly telemetry: 'DISABLED';
  readonly enforcing: boolean;
  readonly blockedAttempts: number;
  readonly allowedConnections: number;
};

export const networkIsolationReport = (
  guard: EgressGuard,
  lanActive: boolean,
): NetworkIsolationReport => ({
  wan: guard.enforcing ? 'BLOCKED' : 'NOT_REQUIRED',
  lan: lanActive ? 'ACTIVE' : 'INACTIVE',
  internetDependency: 'NONE',
  cloudServices: 'DISABLED',
  telemetry: 'DISABLED',
  enforcing: guard.enforcing,
  blockedAttempts: guard.blockedCount,
  allowedConnections: guard.allowedCount,
});
