// Root layout: providers, theme, shell chrome.
import './globals.css';

export const metadata = { title: 'OmniSense', description: 'Autonomous market intelligence' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/* TODO: wrap in QueryProvider + ThemeProvider + AppShell */}
      <body>{children}</body>
    </html>
  );
}
