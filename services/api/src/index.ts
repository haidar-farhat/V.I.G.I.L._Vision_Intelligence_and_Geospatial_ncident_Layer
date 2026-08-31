/**
 * `@sentinel/api` - the control plane.
 *
 * Versioned REST for configuration, a WebSocket hub for realtime, authentication
 * and authorization in front of both. Binds loopback by default: this serves a
 * local desktop application, and a control plane listening on every interface is
 * a decision somebody should have to make deliberately.
 */

export * from './auth.ts';
export * from './router.ts';
export * from './hub.ts';
export * from './server.ts';
