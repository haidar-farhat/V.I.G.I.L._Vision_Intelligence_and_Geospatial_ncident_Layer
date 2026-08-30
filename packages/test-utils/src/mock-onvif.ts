import { createServer } from 'node:http';
import type { Server } from 'node:http';
import { createHash } from 'node:crypto';

/**
 * A mock ONVIF device.
 *
 * Speaks enough of the device and media services to drive a full onboarding
 * flow, and can be told to misbehave in the ways real cameras do: reject a
 * credential the way ONVIF actually signals it (a SOAP fault with HTTP 500,
 * not a 401), return a stream URI containing embedded credentials, report an
 * address only it can reach, or answer with a namespace prefix nobody expects.
 *
 * The embedded-credential case matters most. Real firmware does return
 * `rtsp://user:pass@host/path`, and an integration that passes that string
 * around is how camera passwords reach log files.
 */

export type MockOnvifBehaviour = {
  readonly username?: string;
  readonly password?: string;
  /** Accept any credential, as an unsecured device does. */
  readonly noAuth?: boolean;
  /** Return a stream URI with credentials embedded, as much firmware does. */
  readonly credentialsInStreamUri?: boolean;
  /** Report a stream host the client cannot reach, as a NATed camera does. */
  readonly wrongStreamHost?: string;
  /** Namespace prefix for response elements. Vendors vary wildly. */
  readonly prefix?: string;
  /** Answer with a SOAP fault carrying this text. */
  readonly faultWith?: string;
  /** Include a DTD, which no legitimate device sends. */
  readonly injectDoctype?: boolean;
  /** Serve no media profiles at all. */
  readonly noProfiles?: boolean;
  readonly manufacturer?: string;
  readonly model?: string;
  readonly serialNumber?: string;
};

export type MockOnvifDevice = {
  readonly port: number;
  readonly actions: readonly string[];
  close(): Promise<void>;
};

export const startMockOnvifDevice = async (
  behaviour: MockOnvifBehaviour = {},
): Promise<MockOnvifDevice> => {
  const actions: string[] = [];
  const prefix = behaviour.prefix ?? 'tds';
  const username = behaviour.username ?? 'admin';
  const password = behaviour.password ?? 'correct-horse-battery';

  const envelope = (body: string): string =>
    `${behaviour.injectDoctype === true ? '<!DOCTYPE root [<!ENTITY xxe "test">]>' : ''}` +
    '<?xml version="1.0" encoding="UTF-8"?>' +
    '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">' +
    `<s:Body>${body}</s:Body>` +
    '</s:Envelope>';

  const fault = (text: string): string =>
    envelope(
      '<s:Fault><s:Code><s:Value>s:Sender</s:Value>' +
        '<s:Subcode><s:Value>ter:NotAuthorized</s:Value></s:Subcode></s:Code>' +
        `<s:Reason><s:Text xml:lang="en">${text}</s:Text></s:Reason></s:Fault>`,
    );

  /**
   * Verify a WS-Security UsernameToken PasswordDigest.
   *
   * Base64(SHA1(nonce + created + password)), exactly as the profile specifies -
   * so a client that computes it wrongly fails here rather than on site.
   */
  const authorised = (body: string): boolean => {
    if (behaviour.noAuth === true) return true;

    const digest = /<[^>]*Password[^>]*>([^<]+)<\//i.exec(body)?.[1];
    const nonce = /<[^>]*Nonce[^>]*>([^<]+)<\//i.exec(body)?.[1];
    const created = /<[^>]*Created[^>]*>([^<]+)<\//i.exec(body)?.[1];
    const user = /<[^>]*Username[^>]*>([^<]+)<\//i.exec(body)?.[1];

    if (digest === undefined || nonce === undefined || created === undefined) return false;
    if (user !== username) return false;

    const expected = createHash('sha1')
      .update(Buffer.concat([
        Buffer.from(nonce, 'base64'),
        Buffer.from(created, 'utf8'),
        Buffer.from(password, 'utf8'),
      ]))
      .digest('base64');

    return digest === expected;
  };

  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];

    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8');
      const action = /action="([^"]+)"/.exec(req.headers['content-type'] ?? '')?.[1] ?? '';
      actions.push(action);

      const reply = (status: number, payload: string): void => {
        res.writeHead(status, { 'Content-Type': 'application/soap+xml; charset=utf-8' });
        res.end(payload);
      };

      if (behaviour.faultWith !== undefined) {
        // ONVIF signals faults with HTTP 500, not a 4xx. A client that only
        // inspects the status code learns nothing useful.
        reply(500, fault(behaviour.faultWith));
        return;
      }

      if (!authorised(body)) {
        reply(500, fault('Sender not authorized'));
        return;
      }

      if (action.endsWith('GetDeviceInformation')) {
        reply(
          200,
          envelope(
            `<${prefix}:GetDeviceInformationResponse xmlns:${prefix}="http://www.onvif.org/ver10/device/wsdl">` +
              `<${prefix}:Manufacturer>${behaviour.manufacturer ?? 'Acme Optics'}</${prefix}:Manufacturer>` +
              `<${prefix}:Model>${behaviour.model ?? 'AX-4000'}</${prefix}:Model>` +
              `<${prefix}:FirmwareVersion>2.4.1</${prefix}:FirmwareVersion>` +
              `<${prefix}:SerialNumber>${behaviour.serialNumber ?? 'SN-0424-7781'}</${prefix}:SerialNumber>` +
              `<${prefix}:HardwareId>HW-11</${prefix}:HardwareId>` +
              `</${prefix}:GetDeviceInformationResponse>`,
          ),
        );
        return;
      }

      if (action.endsWith('GetCapabilities')) {
        reply(
          200,
          envelope(
            `<${prefix}:GetCapabilitiesResponse><${prefix}:Capabilities>` +
              '<tt:Media xmlns:tt="http://www.onvif.org/ver10/schema"><tt:XAddr>http://x/media</tt:XAddr></tt:Media>' +
              '<tt:Events xmlns:tt="http://www.onvif.org/ver10/schema"><tt:XAddr>http://x/events</tt:XAddr></tt:Events>' +
              `</${prefix}:Capabilities></${prefix}:GetCapabilitiesResponse>`,
          ),
        );
        return;
      }

      if (action.endsWith('GetProfiles')) {
        if (behaviour.noProfiles === true) {
          reply(200, envelope('<trt:GetProfilesResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl"/>'));
          return;
        }

        // A main stream and a sub stream, which is what nearly every camera offers.
        reply(
          200,
          envelope(
            '<trt:GetProfilesResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">' +
              '<trt:Profiles token="MainProfile" fixed="true">' +
              '<tt:Name>MainStream</tt:Name>' +
              '<tt:VideoEncoderConfiguration token="venc-main">' +
              '<tt:Encoding>H264</tt:Encoding>' +
              '<tt:Resolution><tt:Width>3840</tt:Width><tt:Height>2160</tt:Height></tt:Resolution>' +
              '<tt:RateControl><tt:FrameRateLimit>25</tt:FrameRateLimit><tt:BitrateLimit>8192</tt:BitrateLimit></tt:RateControl>' +
              '<tt:GovLength>50</tt:GovLength>' +
              '</tt:VideoEncoderConfiguration>' +
              '</trt:Profiles>' +
              '<trt:Profiles token="SubProfile" fixed="true">' +
              '<tt:Name>SubStream</tt:Name>' +
              '<tt:VideoEncoderConfiguration token="venc-sub">' +
              '<tt:Encoding>H264</tt:Encoding>' +
              '<tt:Resolution><tt:Width>640</tt:Width><tt:Height>360</tt:Height></tt:Resolution>' +
              '<tt:RateControl><tt:FrameRateLimit>10</tt:FrameRateLimit><tt:BitrateLimit>512</tt:BitrateLimit></tt:RateControl>' +
              '<tt:GovLength>20</tt:GovLength>' +
              '</tt:VideoEncoderConfiguration>' +
              '</trt:Profiles>' +
              '</trt:GetProfilesResponse>',
          ),
        );
        return;
      }

      if (action.endsWith('GetStreamUri')) {
        const token = /<[^>]*ProfileToken[^>]*>([^<]+)<\//i.exec(body)?.[1] ?? 'MainProfile';
        const host = behaviour.wrongStreamHost ?? '127.0.0.1';
        const userinfo = behaviour.credentialsInStreamUri === true ? `${username}:${password}@` : '';
        const path = token === 'SubProfile' ? '/Streaming/Channels/102' : '/Streaming/Channels/101';

        reply(
          200,
          envelope(
            '<trt:GetStreamUriResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">' +
              `<trt:MediaUri><tt:Uri>rtsp://${userinfo}${host}:554${path}</tt:Uri>` +
              '<tt:InvalidAfterConnect>false</tt:InvalidAfterConnect>' +
              '<tt:Timeout>PT60S</tt:Timeout></trt:MediaUri>' +
              '</trt:GetStreamUriResponse>',
          ),
        );
        return;
      }

      reply(500, fault('Optional Action Not Implemented'));
    });
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('mock ONVIF device did not bind a port'));
        return;
      }
      resolve(address.port);
    });
    server.once('error', reject);
  });

  return {
    port,
    actions,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
};

/** A WS-Discovery ProbeMatch as a camera would send it, for parser tests. */
export const probeMatchXml = (options: {
  readonly address?: string;
  readonly xaddrs?: string;
  readonly scopes?: string;
} = {}): string =>
  '<?xml version="1.0" encoding="UTF-8"?>' +
  '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" ' +
  'xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing" ' +
  'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">' +
  '<s:Header><a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</a:Action></s:Header>' +
  '<s:Body><d:ProbeMatches><d:ProbeMatch>' +
  `<a:EndpointReference><a:Address>${options.address ?? 'urn:uuid:11223344-5566-7788-99aa-bbccddeeff00'}</a:Address></a:EndpointReference>` +
  '<d:Types>dn:NetworkVideoTransmitter</d:Types>' +
  `<d:Scopes>${options.scopes ?? 'onvif://www.onvif.org/name/Front%20Door onvif://www.onvif.org/hardware/AX-4000 onvif://www.onvif.org/location/Gate'}</d:Scopes>` +
  `<d:XAddrs>${options.xaddrs ?? 'http://192.168.1.50/onvif/device_service'}</d:XAddrs>` +
  '<d:MetadataVersion>1</d:MetadataVersion>' +
  '</d:ProbeMatch></d:ProbeMatches></s:Body></s:Envelope>';
