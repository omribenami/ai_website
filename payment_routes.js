// Payment Verification Routes for Buy Me a Coffee Integration

// server/routes/payment.js
const express = require('express');
const {
  submitVerification,
  getVerificationStatus,
  getPendingVerifications,
  approveVerification,
  rejectVerification,
  getPaymentInstructions
} = require('../controllers/payment');

const router = express.Router();

// Import middleware
const { protect, authorize } = require('../middleware/auth');

// Routes
router.post('/verify', protect, submitVerification);
router.get('/status', protect, getVerificationStatus);
router.get('/instructions', protect, getPaymentInstructions);
router.get('/pending', protect, authorize('admin'), getPendingVerifications);
router.put('/approve/:id', protect, authorize('admin'), approveVerification);
router.put('/reject/:id', protect, authorize('admin'), rejectVerification);

module.exports = router;
