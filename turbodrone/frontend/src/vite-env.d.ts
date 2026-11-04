/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL?: string;
  readonly VITE_TWIN_CHASE_URL?: string;
  readonly VITE_PACK_WARN_S?: string;
  readonly VITE_PACK_LAND_S?: string;
  readonly VITE_PACKS?: string;
  readonly VITE_AXES_RELEASE_MESSAGE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
