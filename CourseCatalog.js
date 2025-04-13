import React, { useState, useEffect } from 'react';
import { Box, Container, Typography, Grid, Card, CardContent, CardMedia, CardActionArea, Button, TextField, InputAdornment } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

const CourseCatalog = () => {
  const [courses, setCourses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  
  useEffect(() => {
    // In a real implementation, this would fetch from an API
    const fetchCourses = async () => {
      try {
        setLoading(true);
        
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Mock data
        const mockCourses = [
          {
            id: 'introduction-to-ai',
            title: 'Introduction to AI and Machine Learning',
            description: 'Learn the fundamentals of artificial intelligence and machine learning concepts.',
            thumbnail: 'https://via.placeholder.com/300x200?text=AI+Introduction',
            price: 49.99,
            level: 'beginner',
            duration: '10 hours',
            modules: 5,
            rating: 4.7
          },
          {
            id: 'environment-setup',
            title: 'AI Development Environment Setup',
            description: 'Set up your local and cloud environments for AI development with GPU acceleration.',
            thumbnail: 'https://via.placeholder.com/300x200?text=Environment+Setup',
            price: 29.99,
            level: 'beginner',
            duration: '5 hours',
            modules: 3,
            rating: 4.5
          },
          {
            id: 'python-for-ai',
            title: 'Python for AI Development',
            description: 'Master Python programming specifically for AI and machine learning applications.',
            thumbnail: 'https://via.placeholder.com/300x200?text=Python+for+AI',
            price: 59.99,
            level: 'intermediate',
            duration: '15 hours',
            modules: 8,
            rating: 4.8
          },
          {
            id: 'ml-fundamentals',
            title: 'Machine Learning Fundamentals',
            description: 'Understand core machine learning algorithms, workflows, and evaluation metrics.',
            thumbnail: 'https://via.placeholder.com/300x200?text=ML+Fundamentals',
            price: 79.99,
            level: 'intermediate',
            duration: '20 hours',
            modules: 10,
            rating: 4.9
          },
          {
            id: 'deep-learning',
            title: 'Deep Learning Essentials',
            description: 'Build and train neural networks using modern deep learning frameworks.',
            thumbnail: 'https://via.placeholder.com/300x200?text=Deep+Learning',
            price: 99.99,
            level: 'advanced',
            duration: '25 hours',
            modules: 12,
            rating: 4.6
          },
          {
            id: 'practical-ai-projects',
            title: 'Practical AI Projects',
            description: 'Apply your knowledge with hands-on projects in computer vision, NLP, and more.',
            thumbnail: 'https://via.placeholder.com/300x200?text=Practical+Projects',
            price: 129.99,
            level: 'advanced',
            duration: '30 hours',
            modules: 15,
            rating: 4.8
          }
        ];
        
        setCourses(mockCourses);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching courses:', error);
        setLoading(false);
      }
    };
    
    fetchCourses();
  }, []);
  
  const filteredCourses = courses.filter(course => 
    course.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    course.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    course.level.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };
  
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        AI Expert Guide Courses
      </Typography>
      
      <Typography variant="h6" component="h2" gutterBottom align="center" color="text.secondary" sx={{ mb: 4 }}>
        From Zero to Hero: Master AI Development with Hands-on Learning
      </Typography>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
        <TextField
          label="Search Courses"
          variant="outlined"
          value={searchTerm}
          onChange={handleSearchChange}
          sx={{ width: { xs: '100%', sm: '60%', md: '50%' } }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
      </Box>
      
      {loading ? (
        <Typography>Loading courses...</Typography>
      ) : (
        <Grid container spacing={4}>
          {filteredCourses.map((course) => (
            <Grid item key={course.id} xs={12} sm={6} md={4}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardActionArea href={`/courses/${course.id}`}>
                  <CardMedia
                    component="img"
                    height="140"
                    image={course.thumbnail}
                    alt={course.title}
                  />
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography gutterBottom variant="h5" component="h2">
                      {course.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {course.description}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Level: {course.level.charAt(0).toUpperCase() + course.level.slice(1)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {course.duration}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="h6" color="primary">
                        ${course.price}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Rating: {course.rating}/5
                      </Typography>
                    </Box>
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      
      {filteredCourses.length === 0 && !loading && (
        <Box sx={{ textAlign: 'center', my: 4 }}>
          <Typography variant="h6">No courses found matching your search.</Typography>
          <Button variant="contained" onClick={() => setSearchTerm('')} sx={{ mt: 2 }}>
            Clear Search
          </Button>
        </Box>
      )}
    </Container>
  );
};

export default CourseCatalog;
