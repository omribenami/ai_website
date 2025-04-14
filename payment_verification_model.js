// Payment Verification System for Buy Me a Coffee Integration

// server/models/PaymentVerification.js
const mongoose = require('mongoose');

const PaymentVerificationSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  submissionDate: {
    type: Date,
    default: Date.now
  },
  verificationCode: {
    type: String,
    required: true,
    unique: true
  },
  status: {
    type: String,
    enum: ['pending', 'approved', 'rejected'],
    default: 'pending'
  },
  reviewedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  },
  reviewDate: Date,
  notes: String,
  evidence: String // URL to screenshot/receipt
});

module.exports = mongoose.model('PaymentVerification', PaymentVerificationSchema);
