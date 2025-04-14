// Payment Verification Management Component for AI Expert Guide Blog

// client/src/components/admin/PaymentVerification.js
import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Paper, Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, TablePagination, Button, IconButton, Dialog, 
  DialogTitle, DialogContent, DialogActions, TextField, CircularProgress, 
  Snackbar, Alert, Chip, Card, CardMedia, CardContent, Divider
} from '@mui/material';
import { CheckCircle, Cancel, Visibility } from '@mui/icons-material';
import axios from 'axios';

const PaymentVerificationManagement = () => {
  const [verifications, setVerifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [openDialog, setOpenDialog] = useState(false);
  const [currentVerification, setCurrentVerification] = useState(null);
  const [notes, setNotes] = useState('');
  const [viewEvidenceDialog, setViewEvidenceDialog] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  useEffect(() => {
    fetchVerifications();
  }, []);

  const fetchVerifications = async () => {
    try {
      setLoading(true);
      const res = await axios.get('/api/v1/payments/pending');
      setVerifications(res.data.data);
      setLoading(false);
    } catch (err) {
      setError('Error fetching payment verifications');
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

  const handleOpenReviewDialog = (verification) => {
    setCurrentVerification(verification);
    setNotes('');
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleOpenEvidenceDialog = (verification) => {
    setCurrentVerification(verification);
    setViewEvidenceDialog(true);
  };

  const handleCloseEvidenceDialog = () => {
    setViewEvidenceDialog(false);
  };

  const handleApproveVerification = async () => {
    try {
      await axios.put(`/api/v1/payments/approve/${currentVerification._id}`, { notes });
      setSnackbar({
        open: true,
        message: 'Payment verification approved successfully',
        severity: 'success'
      });
      handleCloseDialog();
      fetchVerifications();
    } catch (err) {
      setSnackbar({
        open: true,
        message: `Error: ${err.response?.data?.error || 'Something went wrong'}`,
        severity: 'error'
      });
    }
  };

  const handleRejectVerification = async () => {
    try {
      await axios.put(`/api/v1/payments/reject/${currentVerification._id}`, { notes });
      setSnackbar({
        open: true,
        message: 'Payment verification rejected',
        severity: 'success'
      });
      handleCloseDialog();
      fetchVerifications();
    } catch (err) {
      setSnackbar({
        open: true,
        message: `Error: ${err.response?.data?.error || 'Something went wrong'}`,
        severity: 'error'
      });
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
      <Typography variant="h4" gutterBottom>
        Payment Verification Management
      </Typography>

      {verifications.length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6">No pending payment verifications</Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>User</TableCell>
                <TableCell>Verification Code</TableCell>
                <TableCell>Submission Date</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Evidence</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {verifications
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((verification) => (
                  <TableRow key={verification._id}>
                    <TableCell>
                      {verification.userId ? (
                        <>
                          <Typography variant="body2" fontWeight="bold">
                            {verification.userId.username}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            {verification.userId.email}
                          </Typography>
                        </>
                      ) : (
                        'Unknown User'
                      )}
                    </TableCell>
                    <TableCell>{verification.verificationCode}</TableCell>
                    <TableCell>{new Date(verification.submissionDate).toLocaleString()}</TableCell>
                    <TableCell>
                      <Chip 
                        label={verification.status} 
                        color={
                          verification.status === 'pending' ? 'warning' : 
                          verification.status === 'approved' ? 'success' : 
                          'error'
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {verification.evidence ? (
                        <IconButton 
                          color="primary" 
                          onClick={() => handleOpenEvidenceDialog(verification)}
                        >
                          <Visibility />
                        </IconButton>
                      ) : (
                        'No evidence'
                      )}
                    </TableCell>
                    <TableCell>
                      <Button 
                        variant="contained" 
                        color="primary" 
                        size="small"
                        onClick={() => handleOpenReviewDialog(verification)}
                        sx={{ mr: 1 }}
                      >
                        Review
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={verifications.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </TableContainer>
      )}

      {/* Review Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Review Payment Verification</DialogTitle>
        <DialogContent>
          {currentVerification && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                User: {currentVerification.userId?.username || 'Unknown'}
              </Typography>
              <Typography variant="body2" gutterBottom>
                Email: {currentVerification.userId?.email || 'Unknown'}
              </Typography>
              <Typography variant="body2" gutterBottom>
                Verification Code: {currentVerification.verificationCode}
              </Typography>
              <Typography variant="body2" gutterBottom>
                Submitted: {new Date(currentVerification.submissionDate).toLocaleString()}
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <TextField
                label="Admin Notes"
                multiline
                rows={4}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                fullWidth
                variant="outlined"
                placeholder="Add notes about this verification (optional)"
                sx={{ mb: 2 }}
              />
              
              {currentVerification.evidence && (
                <Button 
                  variant="outlined" 
                  onClick={() => handleOpenEvidenceDialog(currentVerification)}
                  fullWidth
                >
                  View Evidence
                </Button>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button 
            onClick={handleRejectVerification} 
            variant="outlined" 
            color="error"
            startIcon={<Cancel />}
          >
            Reject
          </Button>
          <Button 
            onClick={handleApproveVerification} 
            variant="contained" 
            color="success"
            startIcon={<CheckCircle />}
          >
            Approve
          </Button>
        </DialogActions>
      </Dialog>

      {/* Evidence Dialog */}
      <Dialog open={viewEvidenceDialog} onClose={handleCloseEvidenceDialog} maxWidth="md" fullWidth>
        <DialogTitle>Payment Evidence</DialogTitle>
        <DialogContent>
          {currentVerification && (
            <Card>
              {currentVerification.evidence && (
                <CardMedia
                  component="img"
                  image={currentVerification.evidence}
                  alt="Payment Evidence"
                  sx={{ maxHeight: '70vh', objectFit: 'contain' }}
                />
              )}
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  Submitted by {currentVerification.userId?.username || 'Unknown User'} on {new Date(currentVerification.submissionDate).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseEvidenceDialog}>Close</Button>
          {currentVerification && currentVerification.status === 'pending' && (
            <>
              <Button 
                onClick={handleRejectVerification} 
                variant="outlined" 
                color="error"
              >
                Reject
              </Button>
              <Button 
                onClick={handleApproveVerification} 
                variant="contained" 
                color="success"
              >
                Approve
              </Button>
            </>
          )}
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

export default PaymentVerificationManagement;
