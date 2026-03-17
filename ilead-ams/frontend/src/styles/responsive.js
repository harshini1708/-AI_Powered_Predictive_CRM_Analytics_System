/**
 * Responsive Design Utilities
 * Mobile-first approach
 */

export const breakpoints = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1400
};

/**
 * Media query generator
 */
export const media = {
  xs: (styles) => `@media (min-width: ${breakpoints.xs}px) { ${styles} }`,
  sm: (styles) => `@media (min-width: ${breakpoints.sm}px) { ${styles} }`,
  md: (styles) => `@media (min-width: ${breakpoints.md}px) { ${styles} }`,
  lg: (styles) => `@media (min-width: ${breakpoints.lg}px) { ${styles} }`,
  xl: (styles) => `@media (min-width: ${breakpoints.xl}px) { ${styles} }`,
  xxl: (styles) => `@media (min-width: ${breakpoints.xxl}px) { ${styles} }`,
  mobile: (styles) => `@media (max-width: ${breakpoints.md - 1}px) { ${styles} }`,
  tablet: (styles) => `@media (min-width: ${breakpoints.md}px) and (max-width: ${breakpoints.lg - 1}px) { ${styles} }`,
  desktop: (styles) => `@media (min-width: ${breakpoints.lg}px) { ${styles} }`
};

/**
 * Responsive grid system
 */
export const createResponsiveGrid = (mobileColumns = 1, tabletColumns = 2, desktopColumns = 3) => {
  return {
    display: 'grid',
    gridTemplateColumns: `repeat(${mobileColumns}, 1fr)`,
    gap: '20px',
    '@media (min-width: 768px)': {
      gridTemplateColumns: `repeat(${tabletColumns}, 1fr)`
    },
    '@media (min-width: 992px)': {
      gridTemplateColumns: `repeat(${desktopColumns}, 1fr)`
    }
  };
};

/**
 * Responsive typography
 */
export const responsiveTypography = {
  h1: {
    fontSize: 'clamp(24px, 5vw, 48px)',
    lineHeight: 1.2
  },
  h2: {
    fontSize: 'clamp(20px, 4vw, 36px)',
    lineHeight: 1.3
  },
  h3: {
    fontSize: 'clamp(18px, 3vw, 28px)',
    lineHeight: 1.3
  },
  body: {
    fontSize: 'clamp(14px, 2vw, 16px)',
    lineHeight: 1.5
  },
  small: {
    fontSize: 'clamp(12px, 1.5vw, 14px)',
    lineHeight: 1.4
  }
};

/**
 * Safe area insets (for notches)
 */
export const safeAreaInsets = {
  top: 'max(20px, env(safe-area-inset-top))',
  right: 'max(20px, env(safe-area-inset-right))',
  bottom: 'max(20px, env(safe-area-inset-bottom))',
  left: 'max(20px, env(safe-area-inset-left))'
};

/**
 * Touch-friendly sizing
 */
export const touchTargets = {
  minSize: '44px',
  minGap: '8px'
};

/**
 * Aspect ratio utilities
 */
export const aspectRatios = {
  square: '1 / 1',
  video: '16 / 9',
  golden: '1.618 / 1',
  portrait: '3 / 4',
  wide: '21 / 9'
};
