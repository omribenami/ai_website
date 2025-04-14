// Admin User Management Component for AI Expert Guide Blog

// client/src/components/admin/UserManagement.js
import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Paper, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, TablePagination, Button, IconButton, Dialog, 
  DialogTitle, DialogContent, DialogActions, TextField, FormControl, 
  InputLabel, Select, MenuItem, CircularProgress, Snackbar, Alert
} from '@mui/material';
import { Edit, Delete, Add, CheckCircle, Cancel } from '@mui/icons-material';
import axios from 'axios';

const UserManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [openDialog, setOpenDialog] = useState(false);
  const [dialogMode, setDialogMode] = useState('add'); // 'add' or 'edit'
  const [currentUser, setCurrentUser] = useState(null);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    role: 'user'
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const res = await axios.get('/api/v1/admin/users');
      setUsers(res.data.data);
      setLoading(false);
    } catch (err) {
      setError('Error fetching users');
      setLoading(false);
    }
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleOpenAddDialog = () => {
    setDialogMode('add');
    setFormData({
      username: '',
      email: '',
      password: '',
      role: 'user'
    });
    setOpenDialog(true);
  };

  const handleOpenEditDialog = (user) => {
    setDialogMode('edit');
    setCurrentUser(user);
    setFormData({
      username: user.username,
      email: user.email,
      password: '',
      role: user.role
    });
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async () => {
    try {
      if (dialogMode === 'add') {
        await axios.post('/api/v1/admin/users', formData);
        setSnackbar({
          open: true,
          message: 'User added successfully',
          severity: 'success'
        });
      } else {
        // For edit mode, don't send password if it's empty
        const updateData = { ...formData };
        if (!updateData.password) delete updateData.password;
        
        await axios.put(`/api/v1/admin/users/${currentUser._id}`, updateData);
        setSnackbar({
          open: true,
          message: 'User updated successfully',
          severity: 'success'
        });
      }
      
      handleCloseDialog();
      fetchUsers();
    } catch (err) {
      setSnackbar({
        open: true,
        message: `Error: ${err.response?.data?.error || 'Something went wrong'}`,
        severity: 'error'
      });
    }
  };

  const handleDeleteUser = async (userId) => {
    if (window.confirm('Are you sure you want to delete this user?')) {
      try {
        await axios.delete(`/api/v1/admin/users/${userId}`);
        setSnackbar({
          open: true,
          message: 'User deleted successfully',
          severity: 'success'
        });
        fetchUsers();
      } catch (err) {
        setSnackbar({
          open: true,
          message: `Error: ${err.response?.data?.error || 'Something went wrong'}`,
          severity: 'error'
        });
      }
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({
      ...snackbar,
      open: false
    });
  };

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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">User Management</Typography>
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<Add />}
          onClick={handleOpenAddDialog}
        >
          Add User
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Username</TableCell>
              <TableCell>Email</TableCell>
              <TableCell>Role</TableCell>
              <TableCell>Registration Date</TableCell>
              <TableCell>Last Login</TableCell>
              <TableCell>Payment Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((user) => (
                <TableRow key={user._id}>
                  <TableCell>{user.username}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>{user.role}</TableCell>
                  <TableCell>{new Date(user.registrationDate).toLocaleDateString()}</TableCell>
                  <TableCell>{new Date(user.lastLogin).toLocaleDateString()}</TableCell>
                  <TableCell>
                    {user.payment?.status === 'verified' ? (
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <CheckCircle color="success" sx={{ mr: 1 }} />
                        Verified
                      </Box>
                    ) : user.payment?.status === 'pending' ? (
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <CircularProgress size={20} sx={{ mr: 1 }} />
                        Pending
                      </Box>
                    ) : (
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Cancel color="error" sx={{ mr: 1 }} />
                        None
                      </Box>
                    )}
                  </TableCell>
                  <TableCell>
                    <IconButton color="primary" onClick={() => handleOpenEditDialog(user)}>
                      <Edit />
                    </IconButton>
                    <IconButton color="error" onClick={() => handleDeleteUser(user._id)}>
                      <Delete />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={users.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </TableContainer>

      {/* Add/Edit User Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{dialogMode === 'add' ? 'Add New User' : 'Edit User'}</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              name="username"
              label="Username"
              value={formData.username}
              onChange={handleInputChange}
              fullWidth
              required
            />
            <TextField
              name="email"
              label="Email"
              type="email"
              value={formData.email}
              onChange={handleInputChange}
              fullWidth
              required
            />
            <TextField
              name="password"
              label={dialogMode === 'add' ? "Password" : "New Password (leave blank to keep current)"}
              type="password"
              value={formData.password}
              onChange={handleInputChange}
              fullWidth
              required={dialogMode === 'add'}
            />
            <FormControl fullWidth>
              <InputLabel>Role</InputLabel>
              <Select
                name="role"
                value={formData.role}
                label="Role"
                onChange={handleInputChange}
              >
                <MenuItem value="user">Regular User</MenuItem>
                <MenuItem value="premium">Premium User</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained" color="primary">
            {dialogMode === 'add' ? 'Add User' : 'Update User'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default UserManagement;
