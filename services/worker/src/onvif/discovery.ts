import { createSocket } from 'node:dgram';
import { networkInterfaces } from 'node:os';
import type { DiscoveredCamera, UtcMillis } from '@sentinel/shared-types';
import { utcMillis } from '@sentinel/shared-types';
import { classifyHost, isLocalScope } from '@sentinel/security';
import { ONVIF_NAMESPACES, assertSafeXml, extractValue, extractValues, messageId } from './soap.ts';

/**
 * ONVIF WS-Discovery.
 *
 * Sends a multicast Probe to 239.255.255.250:3702 and collects ProbeMatch
 * responses. This is entirely link-local: the multicast group is not routed off
 * the subnet, there is no registry to consult, and no name is ever resolved
 * externally. Discovery works with the Internet unplugged because it is
 * structurally incapable of leaving the LAN.
 *
 * Two behaviours matter operationally:
 *
 * **Probe on every interface.** A security host is usually multi-homed - a
 * management NIC and a camera VLAN - and a socket bound to the wrong one finds
 * nothing while reporting success, which looks identical to "there are no
 * cameras". Binding each interface separately is what makes the difference
 * visible.
 *
 * **Never trust the response.** A ProbeMatch carries a device-supplied address.
 * A hostile device on the subnet can claim any address it likes, including one
 * off-network, so every XAddr is validated as private before it is offered to an
 * operator.
 */

export const WS_DISCOVERY_ADDRESS = '239.255.255.250';
export const WS_DISCOVERY_PORT = 3702;

/** ONVIF network video transmitters. Anything else on the group is ignored. */
const NVT_TYPE = 'dn:NetworkVideoTransmitter';

export type DiscoveryOptions = {
  /** How long to listen for responses. */
  readonly timeoutMillis?: number;
  /**
   * Probes to send. Devices drop multicast under load, so a single probe
   * routinely misses a camera that is present and working.
   */
  readonly probeCount?: number;
  readonly probeIntervalMillis?: number;
  /** Restrict to specific local addresses. Defaults to every private IPv4 interface. */
  readonly interfaces?: readonly string[];
  readonly now?: () => UtcMillis;
};

const DEFAULTS = {
  timeoutMillis: 4000,
  probeCount: 3,
  probeIntervalMillis: 500,
} as const;

/** Private IPv4 addresses of every non-loopback interface on this host. */
export const localProbeInterfaces = (): readonly string[] => {
  const addresses: string[] = [];

  for (const entries of Object.values(networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family !== 'IPv4' || entry.internal) continue;
      if (!isLocalScope(classifyHost(entry.address))) continue;
      addresses.push(entry.address);
    }
  }
  return addresses;
};

const buildProbe = (): string =>
  '<?xml version="1.0" encoding="UTF-8"?>' +
  `<s:Envelope xmlns:s="${ONVIF_NAMESPACES.soap}" ` +
  `xmlns:a="${ONVIF_NAMESPACES.addressing}" ` +
  `xmlns:d="${ONVIF_NAMESPACES.discovery}" ` +
  'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">' +
  '<s:Header>' +
  `<a:Action>${ONVIF_NAMESPACES.discovery}/Probe</a:Action>` +
  `<a:MessageID>${messageId()}</a:MessageID>` +
  '<a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>' +
  '</s:Header>' +
  `<s:Body><d:Probe><d:Types>${NVT_TYPE}</d:Types></d:Probe></s:Body>` +
  '</s:Envelope>';

export type ProbeMatch = {
  /** Stable device identifier, usually a urn:uuid. */
  readonly address: string;
  /** Device service endpoints the camera advertised, already validated as local. */
  readonly serviceUrls: readonly string[];
  readonly scopes: readonly string[];
  /** The address the datagram actually came from. Trustworthy, unlike XAddrs. */
  readonly sourceHost: string;
};

/**
 * Parse a ProbeMatch.
 *
 * Returns null for anything that is not a usable match rather than throwing: the
 * discovery group carries traffic from printers, media servers and other
 * unrelated devices, and one unparsable datagram must not abort a scan.
 */
export const parseProbeMatch = (xml: string, sourceHost: string): ProbeMatch | null => {
  try {
    assertSafeXml(xml);
  } catch {
    return null;
  }

  if (!/ProbeMatch/i.test(xml)) return null;

  const addressValue = extractValue(xml, 'Address');
  const xaddrs = extractValue(xml, 'XAddrs');
  if (xaddrs === null) return null;

  const scopes = (extractValue(xml, 'Scopes') ?? '')
    .split(/\s+/)
    .filter((scope) => scope !== '');

  // A device can advertise any address it likes, including one that is not its
  // own and not on this network. Only private endpoints are ever offered up.
  const serviceUrls = xaddrs
    .split(/\s+/)
    .filter((url) => url !== '')
    .filter((url) => {
      try {
        return isLocalScope(classifyHost(new URL(url).hostname));
      } catch {
        return false;
      }
    });

  if (serviceUrls.length === 0) return null;

  return {
    address: addressValue ?? sourceHost,
    serviceUrls,
    scopes,
    sourceHost,
  };
};

/**
 * Read the ONVIF scope vocabulary.
 *
 * Scopes are URIs like `onvif://www.onvif.org/name/Front%20Door`. They are the
 * only identifying detail available before authentication, which makes them what
 * an operator uses to tell one unconfigured camera from another.
 */
export const parseScopes = (
  scopes: readonly string[],
): { name?: string; hardware?: string; location?: string } => {
  const result: { name?: string; hardware?: string; location?: string } = {};

  for (const scope of scopes) {
    const match = /^onvif:\/\/www\.onvif\.org\/(name|hardware|location)\/(.+)$/i.exec(scope);
    const key = match?.[1]?.toLowerCase();
    const value = match?.[2];
    if (key === undefined || value === undefined) continue;

    const decoded = safeDecode(value);
    if (key === 'name') result.name = decoded;
    else if (key === 'hardware') result.hardware = decoded;
    else if (key === 'location') result.location = decoded;
  }
  return result;
};

/** A malformed percent-escape in a device-supplied scope must not throw. */
const safeDecode = (value: string): string => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

/**
 * Probe the LAN for ONVIF cameras.
 *
 * Always resolves, never rejects: a host with one unusable interface should still
 * report what the others found. Failures surface as an absence of results plus a
 * warning, because "discovery crashed" and "no cameras present" must not look the
 * same to an operator.
 */
export const discoverCameras = async (
  options: DiscoveryOptions = {},
): Promise<{ readonly cameras: readonly DiscoveredCamera[]; readonly warnings: readonly string[] }> => {
  const timeout = options.timeoutMillis ?? DEFAULTS.timeoutMillis;
  const probeCount = options.probeCount ?? DEFAULTS.probeCount;
  const probeInterval = options.probeIntervalMillis ?? DEFAULTS.probeIntervalMillis;
  const now = options.now ?? (() => utcMillis(Date.now()));

  const addresses = options.interfaces ?? localProbeInterfaces();
  const warnings: string[] = [];

  if (addresses.length === 0) {
    return {
      cameras: [],
      warnings: ['No private IPv4 interface was available to probe from.'],
    };
  }

  // Keyed by device address so the same camera answering on several interfaces,
  // or to several probes, is reported once.
  const matches = new Map<string, ProbeMatch>();

  await Promise.all(
    addresses.map(
      (address) =>
        new Promise<void>((resolve) => {
          const socket = createSocket({ type: 'udp4', reuseAddr: true });
          let finished = false;

          const finish = (): void => {
            if (finished) return;
            finished = true;
            try {
              socket.close();
            } catch {
              // Already closed; nothing to do.
            }
            resolve();
          };

          socket.on('error', (error) => {
            warnings.push(`Probe from ${address} failed: ${error.message}`);
            finish();
          });

          socket.on('message', (payload, rinfo) => {
            const match = parseProbeMatch(payload.toString('utf8'), rinfo.address);
            if (match !== null && !matches.has(match.address)) {
              matches.set(match.address, match);
            }
          });

          socket.bind({ address, port: 0 }, () => {
            try {
              socket.setMulticastTTL(1); // Link-local. Never routed off the subnet.
              socket.setMulticastInterface(address);
            } catch (error) {
              warnings.push(
                `Could not configure multicast on ${address}: ${
                  error instanceof Error ? error.message : String(error)
                }`,
              );
            }

            const probe = Buffer.from(buildProbe(), 'utf8');
            let sent = 0;

            const sendProbe = (): void => {
              if (finished || sent >= probeCount) return;
              sent += 1;
              socket.send(probe, WS_DISCOVERY_PORT, WS_DISCOVERY_ADDRESS, (error) => {
                if (error !== null) {
                  warnings.push(`Probe from ${address} could not be sent: ${error.message}`);
                }
              });
              if (sent < probeCount) setTimeout(sendProbe, probeInterval).unref();
            };

            sendProbe();
            setTimeout(finish, timeout).unref();
          });
        }),
    ),
  );

  const cameras: DiscoveredCamera[] = [];
  for (const match of matches.values()) {
    const scope = parseScopes(match.scopes);
    const first = match.serviceUrls[0];
    if (first === undefined) continue;

    let host = match.sourceHost;
    let port = 80;
    try {
      const url = new URL(first);
      host = url.hostname;
      port = url.port === '' ? (url.protocol === 'https:' ? 443 : 80) : Number.parseInt(url.port, 10);
    } catch {
      warnings.push(`Device ${match.address} advertised an unparsable service address.`);
    }

    cameras.push({
      host,
      port,
      onvif: true,
      // Discovery says nothing about RTSP; that is only known after querying the
      // media service, which requires credentials. Claiming it here would be a guess.
      rtsp: false,
      discoveredAt: now(),
      // Every ONVIF device requires authentication for anything useful. Assuming
      // otherwise would present an unusable camera as ready to add.
      requiresAuthentication: true,
      ...(scope.name === undefined ? {} : { name: scope.name }),
      ...(scope.hardware === undefined ? {} : { model: scope.hardware }),
    });
  }

  return { cameras, warnings };
};
