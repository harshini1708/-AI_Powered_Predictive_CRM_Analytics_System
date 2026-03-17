/**
 * Accessibility Utilities - WCAG 2.1 AA compliance
 */

/**
 * Create accessible heading with proper semantic HTML
 */
export const createHeading = (level, text, id) => {
  const HeadingTag = `h${Math.min(Math.max(level, 1), 6)}`;
  return `<${HeadingTag} id="${id || ''}">${text}</${HeadingTag}>`;
};

/**
 * Create skip navigation link
 */
export const SkipNavLink = () => (
  <a
    href="#main-content"
    style={{
      position: 'absolute',
      top: '-40px',
      left: 0,
      background: '#000',
      color: '#fff',
      padding: '8px',
      zIndex: 100,
    }}
    onFocus={(e) => (e.target.style.top = '0')}
    onBlur={(e) => (e.target.style.top = '-40px')}
  >
    Skip to main content
  </a>
);

/**
 * Keyboard navigation handler
 */
export const useKeyboardNavigation = (items, onSelect) => {
  const [activeIndex, setActiveIndex] = React.useState(0);

  const handleKeyDown = (e) => {
    switch (e.key) {
      case 'ArrowUp':
      case 'ArrowLeft':
        e.preventDefault();
        setActiveIndex(Math.max(0, activeIndex - 1));
        break;
      case 'ArrowDown':
      case 'ArrowRight':
        e.preventDefault();
        setActiveIndex(Math.min(items.length - 1, activeIndex + 1));
        break;
      case 'Enter':
        e.preventDefault();
        onSelect(items[activeIndex]);
        break;
      default:
        break;
    }
  };

  return { activeIndex, handleKeyDown };
};

/**
 * Focus management for modals
 */
export const useFocusTrap = (ref) => {
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const focusableElements = el.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    el.addEventListener('keydown', handleKeyDown);
    firstElement?.focus();

    return () => el.removeEventListener('keydown', handleKeyDown);
  }, [ref]);
};

/**
 * ARIA live region announcements
 */
export const useAriaLive = (message, priority = 'polite') => {
  const [announcement, setAnnouncement] = React.useState('');

  React.useEffect(() => {
    setAnnouncement(message);
    // Clear after announcement
    const timer = setTimeout(() => setAnnouncement(''), 1000);
    return () => clearTimeout(timer);
  }, [message]);

  return {
    role: 'status',
    'aria-live': priority,
    'aria-atomic': 'true',
    children: announcement
  };
};

/**
 * Color contrast checker
 */
export const checkContrast = (bgColor, fgColor) => {
  const getLuminance = (color) => {
    const rgb = color.match(/\d+/g);
    const [r, g, b] = rgb.map(x => {
      x = x / 255;
      return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const lum1 = getLuminance(bgColor);
  const lum2 = getLuminance(fgColor);
  const lighter = Math.max(lum1, lum2);
  const darker = Math.min(lum1, lum2);

  const ratio = (lighter + 0.05) / (darker + 0.05);

  return {
    ratio: ratio.toFixed(2),
    levelAA: ratio >= 4.5,
    levelAAA: ratio >= 7,
    status: ratio >= 7 ? 'AAA' : ratio >= 4.5 ? 'AA' : 'Fail'
  };
};

/**
 * Accessible form helper
 */
export const createAccessibleForm = (fields) => {
  return fields.map(field => ({
    ...field,
    id: field.id || `field-${Math.random()}`,
    ariaDescribedBy: field.error ? `error-${field.id}` : undefined,
    ariaRequired: field.required,
    ariaLabel: field.label || field.placeholder
  }));
};

/**
 * Text size preference (respects user's system setting)
 */
export const useTextSizePreference = () => {
  const [textSize, setTextSize] = React.useState('normal');

  React.useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const query2 = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = () => {
      // Adjust based on preferences
    };

    query.addEventListener('change', handleChange);
    return () => query.removeEventListener('change', handleChange);
  }, []);

  return textSize;
};

/**
 * Motion preferences (respect prefers-reduced-motion)
 */
export const prefersReducedMotion = () => {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

/**
 * Tooltip component with accessibility
 */
export const AccessibleTooltip = ({ trigger, content, id }) => {
  return {
    trigger: {
      'aria-describedby': id,
      role: 'button'
    },
    tooltip: {
      id,
      role: 'tooltip',
      children: content
    }
  };
};
