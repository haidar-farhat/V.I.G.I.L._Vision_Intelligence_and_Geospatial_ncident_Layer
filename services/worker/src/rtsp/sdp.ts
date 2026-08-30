/**
 * SDP (RFC 4566) parsing, limited to what an RTSP DESCRIBE returns.
 *
 * This is untrusted input. It arrives from a device on the network that may be
 * counterfeit, compromised, or simply a cheap camera with a careless firmware
 * author - and the last of those is by far the most common. The parser therefore
 * never throws on malformed input, never allocates unboundedly, and returns what
 * it could understand alongside what it could not.
 *
 * Refusing to parse a stream because one attribute line is malformed would take a
 * working camera offline over a cosmetic defect. Trusting the line would be
 * worse. Reporting both is the only honest option.
 */

/** Hard caps. A hostile or broken device must not be able to exhaust memory. */
const MAX_SDP_BYTES = 64 * 1024;
const MAX_LINES = 2000;
const MAX_MEDIA_SECTIONS = 32;

export type SdpMedia = {
  /** "video", "audio", "application", ... */
  readonly kind: string;
  readonly port: number;
  readonly protocol: string;
  readonly payloadTypes: readonly number[];
  /** Value of a=control, used as the SETUP target. */
  readonly control: string | null;
  /** Parsed from a=rtpmap, e.g. "H264" with a 90000 Hz clock. */
  readonly encoding: string | null;
  readonly clockRate: number | null;
  /** Raw a=fmtp parameters, e.g. sprop-parameter-sets for H.264. */
  readonly formatParameters: Readonly<Record<string, string>>;
  readonly attributes: Readonly<Record<string, string>>;
};

export type SdpSession = {
  readonly version: number | null;
  readonly sessionName: string | null;
  /** Value of a session-level a=control, if present. */
  readonly control: string | null;
  readonly media: readonly SdpMedia[];
  /**
   * Lines that could not be understood. Surfaced in diagnostics rather than
   * discarded, because "the camera works but sends a malformed b= line" is a
   * useful thing for an integrator to know.
   */
  readonly warnings: readonly string[];
};

/** Split `a=key:value` into its parts. A bare `a=recvonly` has no value. */
const splitAttribute = (raw: string): { key: string; value: string } => {
  const colon = raw.indexOf(':');
  return colon === -1
    ? { key: raw.trim(), value: '' }
    : { key: raw.slice(0, colon).trim(), value: raw.slice(colon + 1).trim() };
};

/** Parse `a=fmtp:96 packetization-mode=1;sprop-parameter-sets=Z0IA...` */
const parseFormatParameters = (value: string): Record<string, string> => {
  const parameters: Record<string, string> = {};

  // Drop the leading payload type; the rest is a semicolon-separated list.
  const firstSpace = value.indexOf(' ');
  if (firstSpace === -1) return parameters;

  for (const part of value.slice(firstSpace + 1).split(';')) {
    const equals = part.indexOf('=');
    if (equals === -1) continue;

    const key = part.slice(0, equals).trim();
    if (key === '') continue;
    parameters[key] = part.slice(equals + 1).trim();
  }
  return parameters;
};

export const parseSdp = (raw: string): SdpSession => {
  const warnings: string[] = [];

  if (raw.length > MAX_SDP_BYTES) {
    warnings.push(`SDP truncated at ${MAX_SDP_BYTES} bytes`);
  }
  const text = raw.slice(0, MAX_SDP_BYTES);

  const lines = text.split(/\r?\n/).slice(0, MAX_LINES);
  if (text.split(/\r?\n/).length > MAX_LINES) {
    warnings.push(`SDP had more than ${MAX_LINES} lines; the remainder was ignored`);
  }

  let version: number | null = null;
  let sessionName: string | null = null;
  let sessionControl: string | null = null;

  const media: SdpMedia[] = [];

  // Accumulator for the media section currently being read.
  let current: {
    kind: string;
    port: number;
    protocol: string;
    payloadTypes: number[];
    control: string | null;
    encoding: string | null;
    clockRate: number | null;
    formatParameters: Record<string, string>;
    attributes: Record<string, string>;
  } | null = null;

  const flush = (): void => {
    if (current === null) return;
    if (media.length >= MAX_MEDIA_SECTIONS) {
      warnings.push(`more than ${MAX_MEDIA_SECTIONS} media sections; the remainder was ignored`);
      current = null;
      return;
    }
    media.push({
      kind: current.kind,
      port: current.port,
      protocol: current.protocol,
      payloadTypes: current.payloadTypes,
      control: current.control,
      encoding: current.encoding,
      clockRate: current.clockRate,
      formatParameters: current.formatParameters,
      attributes: current.attributes,
    });
    current = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === '') continue;

    // Every SDP line is "<one-char type>=<value>".
    if (trimmed.length < 2 || trimmed[1] !== '=') {
      warnings.push(`malformed line: ${trimmed.slice(0, 80)}`);
      continue;
    }

    const type = trimmed[0];
    const value = trimmed.slice(2);

    switch (type) {
      case 'v': {
        const parsed = Number.parseInt(value, 10);
        version = Number.isNaN(parsed) ? null : parsed;
        break;
      }

      case 's':
        sessionName = value;
        break;

      case 'm': {
        flush();

        // m=<media> <port> <proto> <fmt> ...
        const parts = value.split(/\s+/);
        const kind = parts[0] ?? '';
        const port = Number.parseInt(parts[1] ?? '0', 10);
        const protocol = parts[2] ?? '';

        const payloadTypes: number[] = [];
        for (const token of parts.slice(3)) {
          const payload = Number.parseInt(token, 10);
          if (!Number.isNaN(payload)) payloadTypes.push(payload);
        }

        if (kind === '') {
          warnings.push(`media line with no type: ${value.slice(0, 80)}`);
          break;
        }

        current = {
          kind,
          port: Number.isNaN(port) ? 0 : port,
          protocol,
          payloadTypes,
          control: null,
          encoding: null,
          clockRate: null,
          formatParameters: {},
          attributes: {},
        };
        break;
      }

      case 'a': {
        const { key, value: attributeValue } = splitAttribute(value);

        if (current === null) {
          // Session-level attribute.
          if (key === 'control') sessionControl = attributeValue;
          break;
        }

        current.attributes[key] = attributeValue;

        if (key === 'control') {
          current.control = attributeValue;
        } else if (key === 'rtpmap') {
          // rtpmap:96 H264/90000
          const match = /^\d+\s+([^/]+)\/(\d+)/.exec(attributeValue);
          if (match?.[1] !== undefined) {
            current.encoding = match[1].toUpperCase();
            const rate = Number.parseInt(match[2] ?? '', 10);
            current.clockRate = Number.isNaN(rate) ? null : rate;
          } else {
            warnings.push(`unparsable rtpmap: ${attributeValue.slice(0, 80)}`);
          }
        } else if (key === 'fmtp') {
          current.formatParameters = parseFormatParameters(attributeValue);
        }
        break;
      }

      // Origin, connection, timing, bandwidth and the rest carry nothing this
      // pipeline needs. Skipped deliberately rather than by omission.
      case 'o':
      case 'c':
      case 't':
      case 'b':
      case 'i':
      case 'u':
      case 'e':
      case 'p':
      case 'z':
      case 'k':
      case 'r':
        break;

      default:
        warnings.push(`unknown line type "${String(type)}"`);
    }
  }

  flush();

  return { version, sessionName, control: sessionControl, media, warnings };
};

/**
 * The video track this pipeline can actually use.
 *
 * Returns null when the description contains no usable video, which is a real
 * outcome: an audio-only stream, or a camera offering only a codec the decoder
 * does not handle. The caller reports that as a specific, fixable problem rather
 * than as a generic connection failure.
 */
export const selectVideoTrack = (
  session: SdpSession,
  supported: readonly string[] = ['H264', 'H265', 'HEVC', 'JPEG', 'MP4V-ES'],
): SdpMedia | null => {
  const video = session.media.filter((m) => m.kind === 'video');
  if (video.length === 0) return null;

  // Prefer a track whose codec is one the decoder understands.
  for (const track of video) {
    if (track.encoding !== null && supported.includes(track.encoding)) return track;
  }

  // A track with no rtpmap is still worth attempting: static payload types are
  // implied by the RTP profile, and some older cameras rely on that.
  const unlabelled = video.find((track) => track.encoding === null);
  return unlabelled ?? null;
};

/**
 * Resolve a media section's control attribute against the request URL.
 *
 * RFC 2326 allows control to be absolute, relative, or the literal "*" meaning
 * the session URL itself. Cameras use all three, and getting this wrong produces
 * a SETUP against a URL the device does not recognise - which most report as a
 * bare 454, giving no clue why.
 */
export const resolveControlUrl = (baseUrl: string, control: string | null): string => {
  if (control === null || control === '' || control === '*') return baseUrl;
  if (/^rtsps?:\/\//i.test(control)) return control;

  const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  const suffix = control.startsWith('/') ? control.slice(1) : control;
  return `${base}/${suffix}`;
};
