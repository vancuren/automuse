/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL?: string;
  readonly VITE_TWIN_CHASE_URL?: string;
  readonly VITE_PACK_WARN_S?: string;
  readonly VITE_PACK_LAND_S?: string;
  readonly VITE_PACKS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
