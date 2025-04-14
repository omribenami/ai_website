// Admin Dashboard Frontend Components for AI Expert Guide Blog

// client/src/components/admin/Dashboard.js
import React, { useState, useEffect } from 'react';
import { Box, Typography, Grid, Paper, Card, CardContent, CardHeader, Divider, List, ListItem, ListItemText, ListItemAvatar, Avatar, Button, CircularProgress } from '@mui/material';
import { Person, School, Payment, Pending, CheckCircle, Cancel, TrendingUp } from '@mui/icons-material';
import { Link } from 'react-router-dom';
import axios from 'axios';

const AdminDashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDashboardStats = async () => {
      try {
        const res = await axios.get('/api/v1/admin/dashboard');
        setStats(res.data.data);
        setLoading(false);
      } catch (err) {
        setError('Error fetching dashboard data');
        setLoading(false);
      }
    };

    fetchDashboardStats();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <Typography color="error" variant="h6">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Admin Dashboard
      </Typography>
      
      {/* User Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 140 }}>
            <Typography variant="h6" color="primary">Total Users</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <Person fontSize="large" color="primary" sx={{ mr: 2 }} />
              <Typography variant="h3">{stats.users.total}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 140 }}>
            <Typography variant="h6" color="secondary">Premium Users</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <School fontSize="large" color="secondary" sx={{ mr: 2 }} />
              <Typography variant="h3">{stats.users.premium}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 140 }}>
            <Typography variant="h6" color="info.main">Regular Users</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <Person fontSize="large" color="info" sx={{ mr: 2 }} />
              <Typography variant="h3">{stats.users.regular}</Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Payment Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 140 }}>
            <Typography variant="h6" color="warning.main">Pending Verifications</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <Pending fontSize="large" color="warning" sx={{ mr: 2 }} />
              <Typography variant="h3">{stats.payments.pending}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 140 }}>
            <Typography variant="h6" color="success.main">Approved Payments</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <CheckCircle fontSize="large" color="success" sx={{ mr: 2 }} />
              <Typography variant="h3">{stats.payments.approved}</Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 140 }}>
            <Typography variant="h6" color="error.main">Rejected Payments</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <Cancel fontSize="large" color="error" sx={{ mr: 2 }} />
              <Typography variant="h3">{stats.payments.rejected}</Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Recent Activity */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Recent Users" />
            <Divider />
            <CardContent>
              <List>
                {stats.users.recent.map((user) => (
                  <ListItem key={user._id} divider>
                    <ListItemAvatar>
                      <Avatar>
                        <Person />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText 
                      primary={user.username} 
                      secondary={`${user.email} - Joined: ${new Date(user.registrationDate).toLocaleDateString()}`} 
                    />
                  </ListItem>
                ))}
              </List>
              <Button 
                component={Link} 
                to="/admin/users" 
                variant="outlined" 
                color="primary" 
                fullWidth 
                sx={{ mt: 2 }}
              >
                View All Users
              </Button>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Recent Payment Verifications" />
            <Divider />
            <CardContent>
              <List>
                {stats.payments.recent.map((verification) => (
                  <ListItem key={verification._id} divider>
                    <ListItemAvatar>
                      <Avatar sx={{ 
                        bgcolor: 
                          verification.status === 'pending' ? 'warning.main' : 
                          verification.status === 'approved' ? 'success.main' : 
                          'error.main' 
                      }}>
                        <Payment />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText 
                      primary={verification.userId ? verification.userId.username : 'Unknown User'} 
                      secondary={`Status: ${verification.status} - Submitted: ${new Date(verification.submissionDate).toLocaleDateString()}`} 
                    />
                  </ListItem>
                ))}
              </List>
              <Button 
                component={Link} 
                to="/admin/payments" 
                variant="outlined" 
                color="primary" 
                fullWidth 
                sx={{ mt: 2 }}
              >
                View All Verifications
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdminDashboard;
