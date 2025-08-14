import { defineConfig } from 'tsup'

export default defineConfig({
  // Entry points
  entry: ['src/index.ts'],
  
  // Output formats
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  
  // Target environments
  target: 'es2020',
  
  // Bundling options
  splitting: false,
  treeshake: true,
  minify: false, // Keep readable for debugging
  
  // External dependencies (not bundled)
  external: ['isomorphic-fetch'],
  
  // Output file names
  outExtension({ format }) {
    return {
      js: format === 'cjs' ? '.js' : '.mjs'
    }
  },
  
  // Banner for output files
  banner: {
    js: '// PCS TypeScript SDK v1.0.0'
  },
  
  // Environment variables
  define: {
    __VERSION__: '"1.0.0"'
  }
})