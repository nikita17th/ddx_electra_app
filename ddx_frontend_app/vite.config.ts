import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    build: {
        outDir: 'dist', // Явное указание выходной папки
        emptyOutDir: true // Очистка папки перед сборкой
    }
});