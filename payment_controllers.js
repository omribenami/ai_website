// Payment Verification Controllers for Buy Me a Coffee Integration

// server/controllers/payment.js
const PaymentVerification = require('../models/PaymentVerification');
const User = require('../models/User');
const ErrorResponse = require('../utils/errorResponse');
const asyncHandler = require('../middleware/async');
const crypto = require('crypto');

// @desc    Submit payment verification
// @route   POST /api/v1/payments/verify
// @access  Private
exports.submitVerification = asyncHandler(async (req, res, next) => {
  // Generate a unique verification code
  const verificationCode = crypto.randomBytes(10).toString('hex');
  
  // Create payment verification request
  const verification = await PaymentVerification.create({
    userId: req.user.id,
    verificationCode,
    evidence: req.body.evidence
  });
  
  // Update user payment status to pending
  await User.findByIdAndUpdate(req.user.id, {
    'payment.status': 'pending',
    'payment.verificationCode': verificationCode,
    'payment.paymentDate': Date.now()
  });
  
  res.status(201).json({
    success: true,
    data: verification,
    message: 'Payment verification submitted successfully. An admin will review your submission shortly.'
  });
});

// @desc    Get user's payment verification status
// @route   GET /api/v1/payments/status
// @access  Private
exports.getVerificationStatus = asyncHandler(async (req, res, next) => {
  const user = await User.findById(req.user.id);
  
  if (!user) {
    return next(new ErrorResponse('User not found', 404));
  }
  
  // Get the latest verification request
  const verification = await PaymentVerification.findOne({ 
    userId: req.user.id 
  }).sort({ submissionDate: -1 });
  
  res.status(200).json({
    success: true,
    data: {
      paymentStatus: user.payment.status,
      verification: verification || null
    }
  });
});

// @desc    Get all pending payment verifications
// @route   GET /api/v1/payments/pending
// @access  Private/Admin
exports.getPendingVerifications = asyncHandler(async (req, res, next) => {
  const verifications = await PaymentVerification.find({ 
    status: 'pending' 
  }).populate({
    path: 'userId',
    select: 'username email'
  });
  
  res.status(200).json({
    success: true,
    count: verifications.length,
    data: verifications
  });
});

// @desc    Approve payment verification
// @route   PUT /api/v1/payments/approve/:id
// @access  Private/Admin
exports.approveVerification = asyncHandler(async (req, res, next) => {
  let verification = await PaymentVerification.findById(req.params.id);
  
  if (!verification) {
    return next(new ErrorResponse(`Verification not found with id of ${req.params.id}`, 404));
  }
  
  // Update verification status
  verification = await PaymentVerification.findByIdAndUpdate(req.params.id, {
    status: 'approved',
    reviewedBy: req.user.id,
    reviewDate: Date.now(),
    notes: req.body.notes || 'Payment approved'
  }, {
    new: true,
    runValidators: true
  });
  
  // Update user role and payment status
  await User.findByIdAndUpdate(verification.userId, {
    role: 'premium',
    'payment.status': 'verified',
    'payment.verificationDate': Date.now()
  });
  
  res.status(200).json({
    success: true,
    data: verification
  });
});

// @desc    Reject payment verification
// @route   PUT /api/v1/payments/reject/:id
// @access  Private/Admin
exports.rejectVerification = asyncHandler(async (req, res, next) => {
  let verification = await PaymentVerification.findById(req.params.id);
  
  if (!verification) {
    return next(new ErrorResponse(`Verification not found with id of ${req.params.id}`, 404));
  }
  
  // Update verification status
  verification = await PaymentVerification.findByIdAndUpdate(req.params.id, {
    status: 'rejected',
    reviewedBy: req.user.id,
    reviewDate: Date.now(),
    notes: req.body.notes || 'Payment rejected'
  }, {
    new: true,
    runValidators: true
  });
  
  // Update user payment status
  await User.findByIdAndUpdate(verification.userId, {
    'payment.status': 'none'
  });
  
  res.status(200).json({
    success: true,
    data: verification
  });
});

// @desc    Generate Buy Me a Coffee verification instructions
// @route   GET /api/v1/payments/instructions
// @access  Private
exports.getPaymentInstructions = asyncHandler(async (req, res, next) => {
  // Generate a unique reference code for this user
  const referenceCode = `AI-${req.user.id.toString().slice(-6)}`;
  
  res.status(200).json({
    success: true,
    data: {
      buyMeCoffeeUrl: 'https://buymeacoffee.com/benamiomrik',
      amount: 10,
      referenceCode,
      instructions: [
        `1. Visit our Buy Me a Coffee page at https://buymeacoffee.com/benamiomrik`,
        `2. Click "Support" and select a $10 contribution`,
        `3. In the message field, include your reference code: ${referenceCode}`,
        `4. Complete the payment process`,
        `5. Take a screenshot of your payment confirmation`,
        `6. Return to this site and submit your verification with the screenshot`
      ]
    }
  });
});
