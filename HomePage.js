import React from 'react';
import { Box, Typography, Button, Container } from '@mui/material';
import { ResponsiveHero } from '../components/ResponsiveElements';
import { ResponsiveGrid, TwoColumnLayout } from '../components/ResponsiveLayout';
import ResponsiveNavbar from '../components/ResponsiveNavbar';
import ResponsiveFooter from '../components/ResponsiveFooter';
import { Link as RouterLink } from 'react-router-dom';

const HomePage = () => {
  return (
    <Box>
      <ResponsiveNavbar isAuthenticated={false} />
      
      <ResponsiveHero
        title="Become an AI Expert"
        subtitle="From Zero to Hero: Master AI Development with Hands-on Learning"
        imageSrc="https://via.placeholder.com/1200x800?text=AI+Learning"
        height="70vh"
      >
        <Box sx={{ mt: 4, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button 
            variant="contained" 
            color="secondary" 
            size="large"
            component={RouterLink}
            to="/courses"
          >
            Explore Courses
          </Button>
          <Button 
            variant="outlined" 
            color="inherit" 
            size="large"
            component={RouterLink}
            to="/about"
          >
            Learn More
          </Button>
        </Box>
      </ResponsiveHero>
      
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Typography variant="h2" align="center" gutterBottom>
          Why Choose Our AI Courses?
        </Typography>
        
        <Typography variant="h5" align="center" color="text.secondary" paragraph>
          Comprehensive learning path designed for beginners to become industry-ready AI experts
        </Typography>
        
        <Box sx={{ mt: 6 }}>
          <ResponsiveGrid>
            {[
              {
                title: 'Beginner Friendly',
                description: 'Start from the basics and build a solid foundation in AI concepts and Python programming.',
                icon: '🚀'
              },
              {
                title: 'Hands-on Projects',
                description: 'Apply your knowledge with practical projects using real-world datasets and scenarios.',
                icon: '💻'
              },
              {
                title: 'GPU Optimized',
                description: 'All courses are designed to leverage your RTX 3080 GPU for efficient model training.',
                icon: '⚡'
              },
              {
                title: 'Industry Relevant',
                description: 'Learn the skills and tools that are in high demand in the AI industry.',
                icon: '🏢'
              }
            ].map((feature, index) => (
              <Box 
                key={index} 
                sx={{ 
                  p: 3, 
                  textAlign: 'center',
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                  boxShadow: 1,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center'
                }}
              >
                <Typography variant="h1" component="div" gutterBottom>
                  {feature.icon}
                </Typography>
                <Typography variant="h5" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {feature.description}
                </Typography>
              </Box>
            ))}
          </ResponsiveGrid>
        </Box>
      </Container>
      
      <Box sx={{ bgcolor: 'grey.100', py: 8 }}>
        <Container maxWidth="lg">
          <Typography variant="h2" align="center" gutterBottom>
            Featured Courses
          </Typography>
          
          <Typography variant="h5" align="center" color="text.secondary" paragraph>
            Start your AI journey with our most popular courses
          </Typography>
          
          <Box sx={{ mt: 6 }}>
            <ResponsiveGrid>
              {[
                {
                  title: 'Introduction to AI',
                  description: 'Learn the fundamentals of artificial intelligence and machine learning concepts.',
                  image: 'https://via.placeholder.com/300x200?text=AI+Introduction',
                  path: '/courses/introduction-to-ai'
                },
                {
                  title: 'Python for AI Development',
                  description: 'Master Python programming specifically for AI and machine learning applications.',
                  image: 'https://via.placeholder.com/300x200?text=Python+for+AI',
                  path: '/courses/python-for-ai'
                },
                {
                  title: 'Machine Learning Fundamentals',
                  description: 'Understand core machine learning algorithms, workflows, and evaluation metrics.',
                  image: 'https://via.placeholder.com/300x200?text=ML+Fundamentals',
                  path: '/courses/ml-fundamentals'
                },
                {
                  title: 'Deep Learning Essentials',
                  description: 'Build and train neural networks using modern deep learning frameworks.',
                  image: 'https://via.placeholder.com/300x200?text=Deep+Learning',
                  path: '/courses/deep-learning'
                }
              ].map((course, index) => (
                <Box 
                  key={index} 
                  component={RouterLink}
                  to={course.path}
                  sx={{ 
                    textDecoration: 'none',
                    color: 'inherit',
                    height: '100%',
                    display: 'block'
                  }}
                >
                  <Box 
                    sx={{ 
                      bgcolor: 'background.paper',
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: 1,
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-5px)',
                        boxShadow: 3,
                      }
                    }}
                  >
                    <Box 
                      component="img"
                      src={course.image}
                      alt={course.title}
                      sx={{ 
                        width: '100%',
                        height: 200,
                        objectFit: 'cover'
                      }}
                    />
                    <Box sx={{ p: 3, flexGrow: 1 }}>
                      <Typography variant="h5" component="h3" gutterBottom>
                        {course.title}
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        {course.description}
                      </Typography>
                    </Box>
                    <Box sx={{ p: 3, pt: 0 }}>
                      <Button 
                        variant="outlined" 
                        color="primary"
                        fullWidth
                      >
                        View Course
                      </Button>
                    </Box>
                  </Box>
                </Box>
              ))}
            </ResponsiveGrid>
          </Box>
          
          <Box sx={{ mt: 6, textAlign: 'center' }}>
            <Button 
              variant="contained" 
              color="primary" 
              size="large"
              component={RouterLink}
              to="/courses"
            >
              View All Courses
            </Button>
          </Box>
        </Container>
      </Box>
      
      <Box sx={{ py: 8 }}>
        <Container maxWidth="lg">
          <TwoColumnLayout>
            <Box>
              <Typography variant="h2" gutterBottom>
                Ready to Start Your AI Journey?
              </Typography>
              
              <Typography variant="h5" color="text.secondary" paragraph>
                Join thousands of students who have transformed their careers with our comprehensive AI courses.
              </Typography>
              
              <Typography variant="body1" paragraph>
                Our courses are designed to take you from a beginner to an industry-ready AI expert. With hands-on projects, interactive coding exercises, and comprehensive assessments, you'll build the skills needed to succeed in the rapidly growing field of artificial intelligence.
              </Typography>
              
              <Box sx={{ mt: 4, display: 'flex', gap: 2 }}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  size="large"
                  component={RouterLink}
                  to="/register"
                >
                  Sign Up Now
                </Button>
                <Button 
                  variant="outlined" 
                  color="primary" 
                  size="large"
                  component={RouterLink}
                  to="/contact"
                >
                  Contact Us
                </Button>
              </Box>
            </Box>
            
            <Box 
              component="img"
              src="https://via.placeholder.com/600x400?text=AI+Learning+Journey"
              alt="AI Learning Journey"
              sx={{ 
                width: '100%',
                height: 'auto',
                borderRadius: 2,
                boxShadow: 3
              }}
            />
          </TwoColumnLayout>
        </Container>
      </Box>
      
      <ResponsiveFooter />
    </Box>
  );
};

export default HomePage;
