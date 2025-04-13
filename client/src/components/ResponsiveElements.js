import React from 'react';
import { Box, Typography, Paper, useTheme, useMediaQuery } from '@mui/material';
import { styled } from '@mui/material/styles';

// Responsive card component that adapts to different screen sizes
const ResponsiveCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: theme.shape.borderRadius * 2,
  transition: 'transform 0.3s ease, box-shadow 0.3s ease',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: theme.shadows[6],
  },
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(2),
  },
}));

// Responsive typography that adjusts font size based on screen size
const ResponsiveHeading = styled(Typography)(({ theme }) => ({
  [theme.breakpoints.down('sm')]: {
    fontSize: '1.5rem',
  },
  [theme.breakpoints.between('sm', 'md')]: {
    fontSize: '2rem',
  },
  [theme.breakpoints.up('md')]: {
    fontSize: '2.5rem',
  },
}));

const ResponsiveSubheading = styled(Typography)(({ theme }) => ({
  [theme.breakpoints.down('sm')]: {
    fontSize: '1.2rem',
  },
  [theme.breakpoints.between('sm', 'md')]: {
    fontSize: '1.5rem',
  },
  [theme.breakpoints.up('md')]: {
    fontSize: '1.8rem',
  },
}));

const ResponsiveText = styled(Typography)(({ theme }) => ({
  [theme.breakpoints.down('sm')]: {
    fontSize: '0.875rem',
  },
  [theme.breakpoints.between('sm', 'md')]: {
    fontSize: '1rem',
  },
  [theme.breakpoints.up('md')]: {
    fontSize: '1.1rem',
  },
}));

// Responsive spacing component
const ResponsiveSpacing = styled(Box)(({ theme, size = 'medium' }) => {
  const spacingMap = {
    small: {
      xs: 1,
      sm: 2,
      md: 3,
    },
    medium: {
      xs: 2,
      sm: 4,
      md: 6,
    },
    large: {
      xs: 4,
      sm: 6,
      md: 8,
    },
  };

  return {
    [theme.breakpoints.down('sm')]: {
      margin: theme.spacing(spacingMap[size].xs),
    },
    [theme.breakpoints.between('sm', 'md')]: {
      margin: theme.spacing(spacingMap[size].sm),
    },
    [theme.breakpoints.up('md')]: {
      margin: theme.spacing(spacingMap[size].md),
    },
  };
});

// Responsive image component
const ResponsiveImage = styled('img')(({ theme }) => ({
  maxWidth: '100%',
  height: 'auto',
  borderRadius: theme.shape.borderRadius,
  [theme.breakpoints.down('sm')]: {
    borderRadius: theme.shape.borderRadius / 2,
  },
}));

// Responsive hero section
const ResponsiveHero = ({ 
  title, 
  subtitle, 
  imageSrc, 
  imageAlt = 'Hero image',
  height = 'auto',
  backgroundColor = 'primary.main',
  textColor = 'white',
  children 
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'md'));
  
  return (
    <Box
      sx={{
        position: 'relative',
        height: height,
        minHeight: isMobile ? '300px' : isTablet ? '400px' : '500px',
        backgroundColor: backgroundColor,
        color: textColor,
        display: 'flex',
        flexDirection: isMobile ? 'column' : 'row',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          flex: '1 1 auto',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          p: isMobile ? 3 : isTablet ? 5 : 8,
          zIndex: 1,
        }}
      >
        <ResponsiveHeading variant="h1" gutterBottom>
          {title}
        </ResponsiveHeading>
        
        <ResponsiveSubheading variant="h4" gutterBottom>
          {subtitle}
        </ResponsiveSubheading>
        
        {children}
      </Box>
      
      {imageSrc && (
        <Box
          sx={{
            flex: isMobile ? '1 1 auto' : '0 0 50%',
            position: isMobile ? 'absolute' : 'relative',
            top: isMobile ? 0 : 'auto',
            left: isMobile ? 0 : 'auto',
            right: isMobile ? 0 : 'auto',
            bottom: isMobile ? 0 : 'auto',
            width: isMobile ? '100%' : 'auto',
            height: isMobile ? '100%' : 'auto',
            opacity: isMobile ? 0.2 : 1,
            zIndex: isMobile ? 0 : 1,
            overflow: 'hidden',
          }}
        >
          <img
            src={imageSrc}
            alt={imageAlt}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
          />
        </Box>
      )}
    </Box>
  );
};

// Example usage component
const ResponsiveElements = () => {
  return (
    <Box>
      <ResponsiveHeading variant="h1">Responsive Heading</ResponsiveHeading>
      <ResponsiveSubheading variant="h2">Responsive Subheading</ResponsiveSubheading>
      <ResponsiveText variant="body1">Responsive text that adjusts based on screen size.</ResponsiveText>
      
      <ResponsiveSpacing size="medium" />
      
      <ResponsiveCard>
        <Typography variant="h5">Responsive Card</Typography>
        <Typography variant="body1">This card adapts to different screen sizes.</Typography>
      </ResponsiveCard>
      
      <ResponsiveSpacing size="medium" />
      
      <ResponsiveImage src="https://via.placeholder.com/800x400" alt="Responsive image" />
    </Box>
  );
};

export {
  ResponsiveCard,
  ResponsiveHeading,
  ResponsiveSubheading,
  ResponsiveText,
  ResponsiveSpacing,
  ResponsiveImage,
  ResponsiveHero,
  ResponsiveElements
};
