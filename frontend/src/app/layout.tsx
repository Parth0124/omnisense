/**
 * Root layout: theme, fonts, providers, chrome.
 *
 * `className="dark"` on `<html>` rather than a theme detector. Dark is the
 * designed default here, and letting a first paint happen in light before a
 * script decides otherwise produces the white flash that every theme-toggle
 * implementation eventually has to work around. The light variables exist in
 * `globals.css` for when a toggle is added; until then there is one theme and it
 * renders correctly on the very first frame.
 *
 * `suppressHydrationWarning` on `<html>` because that is where a future theme
 * script would write, and React would otherwise warn about the attribute it
 * finds versus the one it rendered.
 */
import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { AppShell } from '@/components/layout/app-shell';
import { Providers } from '@/app/providers';

const sans = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

// A mono face for ids, quotes and anything the user might copy. Signal ids are
// opaque strings people paste into a search box, and a proportional font makes
// transcription errors between similar glyphs far more likely.
const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: { default: 'OmniSense', template: '%s · OmniSense' },
  description: 'Autonomous market intelligence with verifiable citations.',
};

export const viewport: Viewport = {
  themeColor: '#171a1f',
  colorScheme: 'dark',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <body className="font-sans antialiased">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
