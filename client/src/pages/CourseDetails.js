import React, { useState } from 'react';
import { Box, Container, Typography, TextField, Button, Grid, Paper, Divider, Link } from '@mui/material';
import { useFormik } from 'formik';
import * as Yup from 'yup';

const CourseDetails = () => {
  const [showPreview, setShowPreview] = useState(false);
  
  // Mock course data - in a real implementation, this would be fetched from an API
  const course = {
    id: 'deep-learning',
    title: 'Deep Learning Essentials',
    description: 'Build and train neural networks using modern deep learning frameworks.',
    longDescription: `
      This comprehensive course takes you from the fundamentals of neural networks to advanced deep learning techniques. 
      You'll learn how to build, train, and optimize various types of neural networks using PyTorch and TensorFlow.
      
      With your RTX 3080 GPU, you'll be able to train sophisticated models efficiently and tackle real-world problems
      in computer vision, natural language processing, and more.
    `,
    thumbnail: 'https://via.placeholder.com/800x400?text=Deep+Learning+Course',
    price: 99.99,
    level: 'advanced',
    duration: '25 hours',
    modules: [
      {
        id: 'module-1',
        title: 'Neural Networks Fundamentals',
        sections: [
          { id: 'section-1-1', title: 'The Building Blocks: Neurons and Perceptrons' },
          { id: 'section-1-2', title: 'Multi-layer Neural Networks' },
          { id: 'section-1-3', title: 'Activation Functions' },
          { id: 'section-1-4', title: 'Backpropagation Algorithm' }
        ]
      },
      {
        id: 'module-2',
        title: 'Introduction to Deep Learning Frameworks',
        sections: [
          { id: 'section-2-1', title: 'PyTorch vs. TensorFlow: Detailed Comparison' },
          { id: 'section-2-2', title: 'Setting Up PyTorch with GPU Support' },
          { id: 'section-2-3', title: 'Setting Up TensorFlow with GPU Support' },
          { id: 'section-2-4', title: 'Basic Operations and Tensor Manipulation' }
        ]
      },
      {
        id: 'module-3',
        title: 'Deep Learning Model Training',
        sections: [
          { id: 'section-3-1', title: 'Loss Functions' },
          { id: 'section-3-2', title: 'Optimizers' },
          { id: 'section-3-3', title: 'Learning Rate Scheduling' },
          { id: 'section-3-4', title: 'Batch Normalization' },
          { id: 'section-3-5', title: 'Regularization Techniques' },
          { id: 'section-3-6', title: 'Training Monitoring and Visualization' }
        ]
      }
    ],
    prerequisites: [
      'Basic understanding of Python programming',
      'Familiarity with machine learning concepts',
      'Understanding of linear algebra and calculus'
    ],
    learningObjectives: [
      'Understand neural network architecture and training process',
      'Build and train deep learning models using PyTorch and TensorFlow',
      'Implement various neural network architectures for different tasks',
      'Optimize model performance using advanced techniques',
      'Deploy deep learning models for real-world applications'
    ],
    instructor: {
      name: 'Dr. Neural Network',
      bio: 'AI researcher and educator with 10+ years of experience in deep learning',
      avatar: 'https://via.placeholder.com/150?text=Instructor'
    },
    reviews: [
      {
        id: 'review-1',
        user: 'TensorFlow Fan',
        rating: 5,
        comment: 'Excellent course! The explanations are clear and the practical examples are very helpful.'
      },
      {
        id: 'review-2',
        user: 'PyTorch Enthusiast',
        rating: 4,
        comment: 'Great content, especially the sections on PyTorch. Would recommend to anyone interested in deep learning.'
      }
    ]
  };
  
  // Form validation schema
  const validationSchema = Yup.object({
    email: Yup.string()
      .email('Invalid email address')
      .required('Email is required'),
    name: Yup.string()
      .required('Name is required')
  });
  
  // Form handling
  const formik = useFormik({
    initialValues: {
      email: '',
      name: ''
    },
    validationSchema: validationSchema,
    onSubmit: (values) => {
      // In a real implementation, this would send the data to an API
      console.log('Form submitted:', values);
      alert(`Thank you, ${values.name}! We'll send a preview to ${values.email}`);
      setShowPreview(true);
    }
  });
  
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Grid container spacing={4}>
        {/* Course Header */}
        <Grid item xs={12}>
          <Box sx={{ position: 'relative', mb: 4 }}>
            <img 
              src={course.thumbnail} 
              alt={course.title} 
              style={{ width: '100%', height: 'auto', borderRadius: '8px' }} 
            />
            <Box 
              sx={{ 
                position: 'absolute', 
                bottom: 0, 
                left: 0, 
                right: 0, 
                p: 3, 
                background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
                borderRadius: '0 0 8px 8px'
              }}
            >
              <Typography variant="h3" component="h1" color="white">
                {course.title}
              </Typography>
              <Typography variant="h6" color="white">
                {course.description}
              </Typography>
            </Box>
          </Box>
        </Grid>
        
        {/* Course Details */}
        <Grid item xs={12} md={8}>
          <Typography variant="h4" component="h2" gutterBottom>
            About This Course
          </Typography>
          <Typography variant="body1" paragraph>
            {course.longDescription}
          </Typography>
          
          <Typography variant="h5" component="h3" gutterBottom sx={{ mt: 4 }}>
            What You'll Learn
          </Typography>
          <Grid container spacing={2} sx={{ mb: 4 }}>
            {course.learningObjectives.map((objective, index) => (
              <Grid item xs={12} sm={6} key={index}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box 
                    sx={{ 
                      width: 24, 
                      height: 24, 
                      borderRadius: '50%', 
                      bgcolor: 'primary.main', 
                      color: 'white', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      mr: 1,
                      fontSize: '0.8rem'
                    }}
                  >
                    {index + 1}
                  </Box>
                  <Typography variant="body1">{objective}</Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
          
          <Typography variant="h5" component="h3" gutterBottom>
            Course Content
          </Typography>
          <Paper sx={{ p: 2, mb: 4 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {course.modules.length} modules • {course.duration} • {course.level} level
            </Typography>
            
            {course.modules.map((module, index) => (
              <Box key={module.id} sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Module {index + 1}: {module.title}
                </Typography>
                <Box sx={{ pl: 2 }}>
                  {module.sections.map((section) => (
                    <Typography key={section.id} variant="body2" sx={{ mb: 1 }}>
                      • {section.title}
                    </Typography>
                  ))}
                </Box>
                {index < course.modules.length - 1 && <Divider sx={{ mt: 2 }} />}
              </Box>
            ))}
          </Paper>
          
          <Typography variant="h5" component="h3" gutterBottom>
            Prerequisites
          </Typography>
          <Box sx={{ mb: 4 }}>
            {course.prerequisites.map((prerequisite, index) => (
              <Typography key={index} variant="body1" sx={{ mb: 1 }}>
                • {prerequisite}
              </Typography>
            ))}
          </Box>
          
          <Typography variant="h5" component="h3" gutterBottom>
            Your Instructor
          </Typography>
          <Box sx={{ display: 'flex', mb: 4 }}>
            <img 
              src={course.instructor.avatar} 
              alt={course.instructor.name} 
              style={{ width: 80, height: 80, borderRadius: '50%', marginRight: 16 }} 
            />
            <Box>
              <Typography variant="h6">{course.instructor.name}</Typography>
              <Typography variant="body2" color="text.secondary">
                {course.instructor.bio}
              </Typography>
            </Box>
          </Box>
          
          <Typography variant="h5" component="h3" gutterBottom>
            Student Reviews
          </Typography>
          <Box sx={{ mb: 4 }}>
            {course.reviews.map((review) => (
              <Paper key={review.id} sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle1">{review.user}</Typography>
                  <Typography variant="body2">Rating: {review.rating}/5</Typography>
                </Box>
                <Typography variant="body2">{review.comment}</Typography>
              </Paper>
            ))}
          </Box>
        </Grid>
        
        {/* Course Purchase */}
        <Grid item xs={12} md={4}>
          <Box component={Paper} sx={{ p: 3, position: 'sticky', top: 20 }}>
            <Typography variant="h4" component="div" gutterBottom>
              ${course.price}
            </Typography>
            
            <Button 
              variant="contained" 
              color="primary" 
              fullWidth 
              size="large"
              sx={{ mb: 2 }}
              href="/checkout"
            >
              Enroll Now
            </Button>
            
            <Button 
              variant="outlined" 
              color="primary" 
              fullWidth 
              sx={{ mb: 3 }}
              onClick={() => setShowPreview(!showPreview)}
            >
              {showPreview ? 'Hide Preview' : 'Preview Course'}
            </Button>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body2" sx={{ mb: 2 }}>
              This course includes:
            </Typography>
            
            <Typography variant="body2" sx={{ mb: 1 }}>
              • {course.duration} of on-demand video
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              • Interactive coding exercises
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              • Access on mobile and desktop
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              • Certificate of completion
            </Typography>
            
            <Divider sx={{ my: 2 }} />
            
            {!showPreview ? (
              <Box component="form" onSubmit={formik.handleSubmit}>
                <Typography variant="body1" gutterBottom>
                  Get a free preview:
                </Typography>
                
                <TextField
                  fullWidth
                  id="name"
                  name="name"
                  label="Your Name"
                  value={formik.values.name}
                  onChange={formik.handleChange}
                  error={formik.touched.name && Boolean(formik.errors.name)}
                  helperText={formik.touched.name && formik.errors.name}
                  margin="normal"
                />
                
                <TextField
                  fullWidth
                  id="email"
                  name="email"
                  label="Your Email"
                  value={formik.values.email}
                  onChange={formik.handleChange}
                  error={formik.touched.email && Boolean(formik.errors.email)}
                  helperText={formik.touched.email && formik.errors.email}
                  margin="normal"
                />
                
                <Button 
                  color="primary" 
                  variant="contained" 
                  fullWidth 
                  type="submit"
                  sx={{ mt: 2 }}
                >
                  Get Free Preview
                </Button>
              </Box>
            ) : (
              <Box>
                <Typography variant="body1" gutterBottom>
                  Preview sent! Check your email.
                </Typography>
                <Typography variant="body2">
                  You can also <Link href="/learn/deep-learning/preview">view the preview online</Link>.
                </Typography>
              </Box>
            )}
          </Box>
        </Grid>
      </Grid>
    </Container>
  );
};

export default CourseDetails;
