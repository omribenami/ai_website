import React from 'react';
import { Box, Container, CssBaseline, useMediaQuery, useTheme } from '@mui/material';
import { styled } from '@mui/material/styles';

// Responsive container component that adapts to different screen sizes
const ResponsiveContainer = styled(Container)(({ theme }) => ({
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(2),
  },
  [theme.breakpoints.between('sm', 'md')]: {
    padding: theme.spacing(3),
  },
  [theme.breakpoints.up('md')]: {
    padding: theme.spacing(4),
  },
}));

// Grid layout that adapts to screen size
const ResponsiveGrid = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: theme.spacing(3),
  [theme.breakpoints.down('sm')]: {
    gridTemplateColumns: '1fr',
  },
  [theme.breakpoints.between('sm', 'md')]: {
    gridTemplateColumns: 'repeat(2, 1fr)',
  },
  [theme.breakpoints.up('md')]: {
    gridTemplateColumns: 'repeat(3, 1fr)',
  },
  [theme.breakpoints.up('lg')]: {
    gridTemplateColumns: 'repeat(4, 1fr)',
  },
}));

// Two-column layout that stacks on mobile
const TwoColumnLayout = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: theme.spacing(3),
  [theme.breakpoints.down('md')]: {
    gridTemplateColumns: '1fr',
  },
  [theme.breakpoints.up('md')]: {
    gridTemplateColumns: '3fr 1fr',
  },
}));

// Three-column layout that adapts to screen size
const ThreeColumnLayout = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: theme.spacing(3),
  [theme.breakpoints.down('sm')]: {
    gridTemplateColumns: '1fr',
  },
  [theme.breakpoints.between('sm', 'lg')]: {
    gridTemplateColumns: '1fr 2fr',
  },
  [theme.breakpoints.up('lg')]: {
    gridTemplateColumns: '1fr 3fr 1fr',
  },
}));

// Responsive padding utility
const ResponsivePadding = styled(Box)(({ theme }) => ({
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(2),
  },
  [theme.breakpoints.between('sm', 'md')]: {
    padding: theme.spacing(3),
  },
  [theme.breakpoints.up('md')]: {
    padding: theme.spacing(4),
  },
}));

// Responsive margin utility
const ResponsiveMargin = styled(Box)(({ theme }) => ({
  [theme.breakpoints.down('sm')]: {
    margin: theme.spacing(2),
  },
  [theme.breakpoints.between('sm', 'md')]: {
    margin: theme.spacing(3),
  },
  [theme.breakpoints.up('md')]: {
    margin: theme.spacing(4),
  },
}));

// Hook to check if the current viewport is mobile
const useIsMobile = () => {
  const theme = useTheme();
  return useMediaQuery(theme.breakpoints.down('sm'));
};

// Hook to check if the current viewport is tablet
const useIsTablet = () => {
  const theme = useTheme();
  return useMediaQuery(theme.breakpoints.between('sm', 'md'));
};

// Hook to check if the current viewport is desktop
const useIsDesktop = () => {
  const theme = useTheme();
  return useMediaQuery(theme.breakpoints.up('md'));
};

// Example usage component
const ResponsiveLayout = ({ children }) => {
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const isDesktop = useIsDesktop();

  return (
    <Box>
      <CssBaseline />
      <ResponsiveContainer maxWidth="lg">
        {children}
      </ResponsiveContainer>
    </Box>
  );
};

export {
  ResponsiveContainer,
  ResponsiveGrid,
  TwoColumnLayout,
  ThreeColumnLayout,
  ResponsivePadding,
  ResponsiveMargin,
  useIsMobile,
  useIsTablet,
  useIsDesktop,
  ResponsiveLayout
};
