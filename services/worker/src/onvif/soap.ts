import { createHash, randomBytes, randomUUID } from 'node:crypto';
import type { Secret } from '@sentinel/security';

/**
 * SOAP and WS-Security, scoped to what ONVIF actually needs.
 *
 * **On not using an XML library.** ONVIF responses arrive from devices on the
 * network that may be counterfeit or compromised, and a general-purpose XML
 * parser is a liability there: XXE, billion-laughs entity expansion and DTD
 * fetches are all reachable from parsing untrusted XML with a permissive parser,
 * and the ONVIF surface this system needs is a few dozen scalar fields.
 *
 * So this is a bounded, targeted extractor rather than a parser. It never
 * resolves an entity, never follows a DTD, never expands a reference and never
 * makes a network request - because it has no machinery to do any of those
 * things. The trade is that it cannot handle arbitrary XML, which is exactly the
 * capability that would make it dangerous.
 */

const MAX_RESPONSE_BYTES = 512 * 1024;

export const ONVIF_NAMESPACES = {
  soap: 'http://www.w3.org/2003/05/soap-envelope',
  wsse: 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd',
  wsu: 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd',
  passwordDigest:
    'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest',
  base64:
    'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary',
  device: 'http://www.onvif.org/ver10/device/wsdl',
  media: 'http://www.onvif.org/ver10/media/wsdl',
  media2: 'http://www.onvif.org/ver20/media/wsdl',
  discovery: 'http://schemas.xmlsoap.org/ws/2005/04/discovery',
  addressing: 'http://schemas.xmlsoap.org/ws/2004/08/addressing',
} as const;

/**
 * A WS-Security UsernameToken using PasswordDigest.
 *
 * `Base64(SHA1(nonce + created + password))`. The digest is what goes on the
 * wire; the password itself never leaves this function, and the nonce and
 * timestamp are what stop a captured token from being replayed indefinitely.
 *
 * SHA-1 here is not a choice - it is what the WS-Security UsernameToken profile
 * specifies and what every ONVIF device implements. It is a replay-limited
 * authentication token rather than a signature, so its collision weakness is not
 * the relevant property, but it is worth knowing why it appears in a codebase
 * that otherwise uses SHA-256.
 */
export const buildSecurityHeader = (
  username: string,
  password: Secret<string>,
  now: Date = new Date(),
  nonceBytes: Buffer = randomBytes(16),
): string => {
  const created = now.toISOString().replace(/\.(\d{3})Z$/, '.$1Z');

  const digest = createHash('sha1')
    .update(Buffer.concat([nonceBytes, Buffer.from(created, 'utf8'), Buffer.from(password.expose(), 'utf8')]))
    .digest('base64');

  return (
    `<wsse:Security xmlns:wsse="${ONVIF_NAMESPACES.wsse}" xmlns:wsu="${ONVIF_NAMESPACES.wsu}">` +
    '<wsse:UsernameToken>' +
    `<wsse:Username>${escapeXml(username)}</wsse:Username>` +
    `<wsse:Password Type="${ONVIF_NAMESPACES.passwordDigest}">${digest}</wsse:Password>` +
    `<wsse:Nonce EncodingType="${ONVIF_NAMESPACES.base64}">${nonceBytes.toString('base64')}</wsse:Nonce>` +
    `<wsu:Created>${created}</wsu:Created>` +
    '</wsse:UsernameToken>' +
    '</wsse:Security>'
  );
};

export type SoapCredentials = {
  readonly username: string;
  readonly password: Secret<string>;
};

/** Wrap a body in a SOAP 1.2 envelope, with a security header when credentialed. */
export const buildEnvelope = (
  body: string,
  credentials?: SoapCredentials,
  extraHeader = '',
): string => {
  const security =
    credentials === undefined
      ? ''
      : buildSecurityHeader(credentials.username, credentials.password);

  const header = security === '' && extraHeader === '' ? '' : `<s:Header>${security}${extraHeader}</s:Header>`;

  return (
    '<?xml version="1.0" encoding="UTF-8"?>' +
    `<s:Envelope xmlns:s="${ONVIF_NAMESPACES.soap}">` +
    header +
    `<s:Body>${body}</s:Body>` +
    '</s:Envelope>'
  );
};

/** Minimal XML escaping for values this code puts *into* a request. */
export const escapeXml = (value: string): string =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');

/**
 * Decode the five predefined XML entities, and nothing else.
 *
 * Deliberately does not handle custom entities, parameter entities, or numeric
 * references beyond the basics. A response relying on those is either exotic
 * enough to be worth failing on, or hostile.
 */
const decodeXml = (value: string): string =>
  value
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d{1,7});/g, (_m, code: string) => String.fromCodePoint(Number(code)))
    .replace(/&amp;/g, '&');

/**
 * Reject input that has no business being parsed.
 *
 * A doctype or entity declaration in an ONVIF response is not a formatting
 * quirk - no legitimate device sends one, and both are the entry point for entity
 * expansion attacks. Refusing outright is safer and simpler than sanitising.
 */
export const assertSafeXml = (xml: string): void => {
  if (xml.length > MAX_RESPONSE_BYTES) {
    throw new SoapError(
      `Response exceeded ${MAX_RESPONSE_BYTES} bytes.`,
      'SOAP_RESPONSE_TOO_LARGE',
      false,
    );
  }
  if (/<!DOCTYPE/i.test(xml) || /<!ENTITY/i.test(xml)) {
    throw new SoapError(
      'Response contained a DTD or entity declaration, which no ONVIF device sends and which is refused.',
      'SOAP_UNSAFE_XML',
      false,
    );
  }
};

/**
 * First text value of an element, ignoring namespace prefix.
 *
 * Prefixes vary wildly between vendors - `tds:Manufacturer`, `s0:Manufacturer`,
 * bare `Manufacturer` - and matching on the local name is what makes one code
 * path work across devices.
 */
export const extractValue = (xml: string, localName: string): string | null => {
  const pattern = new RegExp(
    `<(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}(?:\\s[^>]*)?>([\\s\\S]*?)</(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}>`,
    'i',
  );
  const match = pattern.exec(xml);
  return match?.[1] === undefined ? null : decodeXml(match[1].trim());
};

/** Every text value of a repeated element. */
export const extractValues = (xml: string, localName: string): readonly string[] => {
  const pattern = new RegExp(
    `<(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}(?:\\s[^>]*)?>([\\s\\S]*?)</(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}>`,
    'gi',
  );
  const values: string[] = [];
  for (const match of xml.matchAll(pattern)) {
    if (match[1] !== undefined) values.push(decodeXml(match[1].trim()));
  }
  return values;
};

/** Every occurrence of an element including its markup, for nested extraction. */
export const extractBlocks = (xml: string, localName: string): readonly string[] => {
  const pattern = new RegExp(
    `<(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}(?:\\s[^>]*)?>[\\s\\S]*?</(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}>`,
    'gi',
  );
  return [...xml.matchAll(pattern)].map((m) => m[0]);
};

/** Value of an attribute on the first matching element. */
export const extractAttribute = (
  xml: string,
  localName: string,
  attribute: string,
): string | null => {
  const pattern = new RegExp(
    `<(?:[A-Za-z0-9_.-]+:)?${escapeRegex(localName)}\\s[^>]*${escapeRegex(attribute)}="([^"]*)"`,
    'i',
  );
  const match = pattern.exec(xml);
  return match?.[1] === undefined ? null : decodeXml(match[1]);
};

const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export class SoapError extends Error {
  readonly code: string;
  readonly recoverable: boolean;
  /** SOAP fault detail, when the device supplied one. */
  readonly fault: string | null;

  constructor(message: string, code: string, recoverable = true, fault: string | null = null) {
    super(message);
    this.name = 'SoapError';
    this.code = code;
    this.recoverable = recoverable;
    this.fault = fault;
  }
}

/**
 * Turn a SOAP fault into a message an integrator can act on.
 *
 * ONVIF devices bury the useful part several layers down and every vendor picks a
 * different layer, so all the plausible ones are checked. A bare "SOAP fault" with
 * no detail sends somebody hunting through packet captures.
 */
export const faultFrom = (xml: string): string | null => {
  if (!/<(?:[A-Za-z0-9_.-]+:)?Fault[\s>]/i.test(xml)) return null;

  const subcode = extractValue(xml, 'Subcode');
  const value = subcode === null ? null : extractValue(subcode, 'Value');

  return (
    extractValue(xml, 'Text') ??
    value ??
    extractValue(xml, 'Value') ??
    extractValue(xml, 'faultstring') ??
    'the device reported a SOAP fault with no description'
  );
};

/** A message id for WS-Addressing. */
export const messageId = (): string => `urn:uuid:${randomUUID()}`;
