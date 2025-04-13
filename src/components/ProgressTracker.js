import React, { useState, useEffect } from 'react';
import { Box, Typography, LinearProgress, Paper, Divider, Grid, Card, CardContent, Chip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import PauseCircleOutlineIcon from '@mui/icons-material/PauseCircleOutline';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

const ProgressTracker = ({
  courseId,
  userId,
  onProgressUpdate = () => {}
}) => {
  const [progress, setProgress] = useState({
    overall: 0,
    modules: [],
    lastAccessed: null,
    quizScores: [],
    timeSpent: 0,
    certificateEligible: false
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // In a real implementation, this would fetch from an API
    const fetchProgress = async () => {
      try {
        setLoading(true);
        
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Mock data
        const mockProgress = {
          overall: 42, // percentage
          modules: [
            { id: 'module-1', title: 'Introduction to AI', progress: 100, completed: true },
            { id: 'module-2', title: 'Environment Setup', progress: 85, completed: false },
            { id: 'module-3', title: 'Python for AI', progress: 60, completed: false },
            { id: 'module-4', title: 'Machine Learning Fundamentals', progress: 25, completed: false },
            { id: 'module-5', title: 'Deep Learning', progress: 0, completed: false }
          ],
          lastAccessed: new Date(Date.now() - 86400000), // yesterday
          quizScores: [
            { id: 'quiz-1', title: 'AI Concepts Quiz', score: 90, maxScore: 100, completed: true },
            { id: 'quiz-2', title: 'Environment Setup Quiz', score: 75, maxScore: 100, completed: true }
          ],
          timeSpent: 12600, // seconds (3.5 hours)
          certificateEligible: false
        };
        
        setProgress(mockProgress);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching progress:', error);
        setError('Failed to load progress data');
        setLoading(false);
      }
    };
    
    fetchProgress();
  }, [courseId, userId]);

  // Format time spent in hours and minutes
  const formatTimeSpent = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    return `${hours}h ${minutes}m`;
  };

  // Format date to readable string
  const formatDate = (date) => {
    if (!date) return 'Never';
    
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Loading your progress...
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ mb: 4 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Your Course Progress
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body1" fontWeight="medium">
              Overall Completion: {progress.overall}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <AccessTimeIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
              Time Spent: {formatTimeSpent(progress.timeSpent)}
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={progress.overall} 
            sx={{ height: 10, borderRadius: 5 }}
          />
        </Box>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Last Accessed
                </Typography>
                <Typography variant="body1">
                  {formatDate(progress.lastAccessed)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Certificate Status
                </Typography>
                <Typography variant="body1">
                  {progress.certificateEligible ? (
                    <Chip 
                      icon={<CheckCircleIcon />} 
                      label="Eligible for Certificate" 
                      color="success" 
                      size="small" 
                    />
                  ) : (
                    <Chip 
                      icon={<PauseCircleOutlineIcon />} 
                      label="Complete course to earn certificate" 
                      color="default" 
                      size="small" 
                    />
                  )}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        <Typography variant="h6" gutterBottom>
          Module Progress
        </Typography>
        
        {progress.modules.map((module) => (
          <Box key={module.id} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body1">
                {module.title}
                {module.completed && (
                  <CheckCircleIcon 
                    color="success" 
                    fontSize="small" 
                    sx={{ ml: 1, verticalAlign: 'middle' }} 
                  />
                )}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {module.progress}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={module.progress} 
              sx={{ 
                height: 8, 
                borderRadius: 4,
                bgcolor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  bgcolor: module.completed ? 'success.main' : 'primary.main',
                }
              }}
            />
          </Box>
        ))}
        
        <Divider sx={{ my: 3 }} />
        
        <Typography variant="h6" gutterBottom>
          Quiz Performance
        </Typography>
        
        {progress.quizScores.length > 0 ? (
          <Grid container spacing={2}>
            {progress.quizScores.map((quiz) => (
              <Grid item xs={12} sm={6} md={4} key={quiz.id}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                      {quiz.title}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Typography variant="h6" color={quiz.score >= 70 ? 'success.main' : 'warning.main'}>
                        {quiz.score}%
                      </Typography>
                      <Chip 
                        size="small" 
                        label={quiz.score >= 70 ? 'Passed' : 'Needs Review'} 
                        color={quiz.score >= 70 ? 'success' : 'warning'} 
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        ) : (
          <Typography variant="body2" color="text.secondary">
            You haven't completed any quizzes yet.
          </Typography>
        )}
        
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
          <Chip 
            icon={<PlayCircleOutlineIcon />} 
            label="Continue Learning" 
            color="primary" 
            clickable
            onClick={() => {
              // In a real implementation, this would navigate to the last accessed section
              console.log('Continue learning clicked');
            }}
          />
        </Box>
      </Paper>
    </Box>
  );
};

export default ProgressTracker;
