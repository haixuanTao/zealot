import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'zealot',
  tagline: 'GPU-native robot learning in Rust — trained and simulated on nexus',
  favicon: 'img/favicon.png',

  future: {
    v4: true,
  },

  // GitHub Pages project site: https://haixuantao.github.io/zealot/
  url: 'https://haixuantao.github.io',
  baseUrl: '/zealot/',

  organizationName: 'haixuanTao',
  projectName: 'zealot',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: false,
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/nexus-logo.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      items: [
        {
          href: 'https://nexus.dimforge.com',
          label: 'nexus',
          position: 'right',
        },
        {
          href: 'https://github.com/haixuanTao/zealot',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Stack',
          items: [
            {
              label: 'nexus — GPU physics',
              href: 'https://nexus.dimforge.com',
            },
            {
              label: 'Rust-GPU',
              href: 'https://github.com/Rust-GPU/rust-gpu',
            },
            {
              label: 'rapier',
              href: 'https://rapier.rs',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/haixuanTao/zealot',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} zealot. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
