/**
 * Animation and Transition Utilities
 * Smooth, performant animations
 */

export const animations = {
  // Fade animations
  fadeIn: {
    animation: 'fadeIn 0.3s ease-in',
    '@keyframes fadeIn': {
      from: { opacity: 0 },
      to: { opacity: 1 }
    }
  },
  fadeOut: {
    animation: 'fadeOut 0.3s ease-out',
    '@keyframes fadeOut': {
      from: { opacity: 1 },
      to: { opacity: 0 }
    }
  },

  // Slide animations
  slideInLeft: {
    animation: 'slideInLeft 0.3s ease-out',
    '@keyframes slideInLeft': {
      from: { transform: 'translateX(-20px)', opacity: 0 },
      to: { transform: 'translateX(0)', opacity: 1 }
    }
  },
  slideInRight: {
    animation: 'slideInRight 0.3s ease-out',
    '@keyframes slideInRight': {
      from: { transform: 'translateX(20px)', opacity: 0 },
      to: { transform: 'translateX(0)', opacity: 1 }
    }
  },
  slideInUp: {
    animation: 'slideInUp 0.3s ease-out',
    '@keyframes slideInUp': {
      from: { transform: 'translateY(20px)', opacity: 0 },
      to: { transform: 'translateY(0)', opacity: 1 }
    }
  },

  // Scale animations
  scaleIn: {
    animation: 'scaleIn 0.3s ease-out',
    '@keyframes scaleIn': {
      from: { transform: 'scale(0.95)', opacity: 0 },
      to: { transform: 'scale(1)', opacity: 1 }
    }
  },
  pulse: {
    animation: 'pulse 2s infinite',
    '@keyframes pulse': {
      '0%, 100%': { opacity: 1 },
      '50%': { opacity: 0.5 }
    }
  },

  // Bounce animations
  bounce: {
    animation: 'bounce 0.5s ease-in-out',
    '@keyframes bounce': {
      '0%, 100%': { transform: 'translateY(0)' },
      '50%': { transform: 'translateY(-10px)' }
    }
  },

  // Spin animation
  spin: {
    animation: 'spin 1s linear infinite',
    '@keyframes spin': {
      from: { transform: 'rotate(0deg)' },
      to: { transform: 'rotate(360deg)' }
    }
  }
};

/**
 * Transition utilities
 */
export const transitions = {
  fast: 'all 0.15s ease',
  base: 'all 0.2s ease',
  slow: 'all 0.3s ease',
  slower: 'all 0.5s ease',

  // Property-specific
  colors: 'color 0.2s ease, background-color 0.2s ease, border-color 0.2s ease',
  transform: 'transform 0.2s ease',
  opacity: 'opacity 0.2s ease',
  shadow: 'box-shadow 0.2s ease'
};

/**
 * CSS animation injection
 */
export const injectAnimations = () => {
  const style = document.createElement('style');
  let css = '';

  Object.values(animations).forEach((anim) => {
    Object.entries(anim).forEach(([key, value]) => {
      if (key.startsWith('@')) {
        const keyframes = key.replace('@keyframes ', '');
        css += `@keyframes ${keyframes} {`;
        Object.entries(value).forEach(([k, v]) => {
          css += `${k} {`;
          Object.entries(v).forEach(([prop, val]) => {
            css += `${prop}: ${val};`;
          });
          css += `}`;
        });
        css += `}`;
      }
    });
  });

  style.textContent = css;
  document.head.appendChild(style);
};

/**
 * Stagger animation for lists
 */
export const createStaggerAnimation = (index, itemCount, delay = 0.1) => {
  return {
    animation: `slideInUp 0.3s ease-out ${index * delay}s both`,
    '@keyframes slideInUp': {
      from: { transform: 'translateY(20px)', opacity: 0 },
      to: { transform: 'translateY(0)', opacity: 1 }
    }
  };
};

/**
 * Disable animations based on user preference
 */
export const respectReducedMotion = (styles) => {
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    return {
      ...styles,
      animation: 'none',
      transition: 'none'
    };
  }
  return styles;
};

/**
 * Custom hook for animations
 */
export const useAnimation = (animationName, duration = 300) => {
  const [isAnimating, setIsAnimating] = React.useState(false);

  const trigger = () => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), duration);
  };

  return {
    isAnimating,
    trigger,
    animationProps: isAnimating ? { animation: `${animationName} ${duration}ms ease-out` } : {}
  };
};
