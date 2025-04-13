import React from 'react';
import { Box, Typography, Container, Link, Grid, Divider, IconButton, useTheme, useMediaQuery } from '@mui/material';
import FacebookIcon from '@mui/icons-material/Facebook';
import TwitterIcon from '@mui/icons-material/Twitter';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import GitHubIcon from '@mui/icons-material/GitHub';
import YouTubeIcon from '@mui/icons-material/YouTube';
import { Link as RouterLink } from 'react-router-dom';

const ResponsiveFooter = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'md'));
  
  return (
    <Box
      component="footer"
      sx={{
        py: 6,
        px: 2,
        mt: 'auto',
        backgroundColor: theme.palette.grey[100],
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              AI Expert Guide
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Your comprehensive resource for learning AI development from zero to hero.
            </Typography>
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              <IconButton size="small" color="primary" aria-label="facebook">
                <FacebookIcon />
              </IconButton>
              <IconButton size="small" color="primary" aria-label="twitter">
                <TwitterIcon />
              </IconButton>
              <IconButton size="small" color="primary" aria-label="linkedin">
                <LinkedInIcon />
              </IconButton>
              <IconButton size="small" color="primary" aria-label="github">
                <GitHubIcon />
              </IconButton>
              <IconButton size="small" color="primary" aria-label="youtube">
                <YouTubeIcon />
              </IconButton>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              Courses
            </Typography>
            <Link component={RouterLink} to="/courses/introduction-to-ai" color="inherit" display="block" sx={{ mb: 1 }}>
              Introduction to AI
            </Link>
            <Link component={RouterLink} to="/courses/environment-setup" color="inherit" display="block" sx={{ mb: 1 }}>
              Environment Setup
            </Link>
            <Link component={RouterLink} to="/courses/python-for-ai" color="inherit" display="block" sx={{ mb: 1 }}>
              Python for AI
            </Link>
            <Link component={RouterLink} to="/courses/ml-fundamentals" color="inherit" display="block" sx={{ mb: 1 }}>
              ML Fundamentals
            </Link>
            <Link component={RouterLink} to="/courses/deep-learning" color="inherit" display="block" sx={{ mb: 1 }}>
              Deep Learning
            </Link>
            <Link component={RouterLink} to="/courses" color="primary" display="block" sx={{ mb: 1 }}>
              View All Courses
            </Link>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              Resources
            </Typography>
            <Link component={RouterLink} to="/blog" color="inherit" display="block" sx={{ mb: 1 }}>
              Blog
            </Link>
            <Link component={RouterLink} to="/tutorials" color="inherit" display="block" sx={{ mb: 1 }}>
              Tutorials
            </Link>
            <Link component={RouterLink} to="/documentation" color="inherit" display="block" sx={{ mb: 1 }}>
              Documentation
            </Link>
            <Link component={RouterLink} to="/faq" color="inherit" display="block" sx={{ mb: 1 }}>
              FAQ
            </Link>
            <Link component={RouterLink} to="/community" color="inherit" display="block" sx={{ mb: 1 }}>
              Community
            </Link>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              Company
            </Typography>
            <Link component={RouterLink} to="/about" color="inherit" display="block" sx={{ mb: 1 }}>
              About Us
            </Link>
            <Link component={RouterLink} to="/contact" color="inherit" display="block" sx={{ mb: 1 }}>
              Contact
            </Link>
            <Link component={RouterLink} to="/careers" color="inherit" display="block" sx={{ mb: 1 }}>
              Careers
            </Link>
            <Link component={RouterLink} to="/terms" color="inherit" display="block" sx={{ mb: 1 }}>
              Terms of Service
            </Link>
            <Link component={RouterLink} to="/privacy" color="inherit" display="block" sx={{ mb: 1 }}>
              Privacy Policy
            </Link>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 4 }} />
        
        <Box sx={{ 
          display: 'flex', 
          flexDirection: isMobile ? 'column' : 'row',
          justifyContent: 'space-between',
          alignItems: isMobile ? 'center' : 'flex-start',
          textAlign: isMobile ? 'center' : 'left',
          gap: 2
        }}>
          <Typography variant="body2" color="text.secondary">
            © {new Date().getFullYear()} AI Expert Guide. All rights reserved.
          </Typography>
          
          <Box sx={{ 
            display: 'flex', 
            flexDirection: isMobile ? 'column' : 'row',
            gap: isMobile ? 1 : 3,
            alignItems: 'center'
          }}>
            <Link component={RouterLink} to="/terms" color="inherit" variant="body2">
              Terms
            </Link>
            <Link component={RouterLink} to="/privacy" color="inherit" variant="body2">
              Privacy
            </Link>
            <Link component={RouterLink} to="/cookies" color="inherit" variant="body2">
              Cookies
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default ResponsiveFooter;
