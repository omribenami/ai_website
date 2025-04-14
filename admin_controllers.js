// Admin Interface Components for AI Expert Guide Blog

// server/controllers/admin.js
const User = require('../models/User');
const PaymentVerification = require('../models/PaymentVerification');
const Content = require('../models/Content');
const ErrorResponse = require('../utils/errorResponse');
const asyncHandler = require('../middleware/async');

// @desc    Get all users
// @route   GET /api/v1/admin/users
// @access  Private/Admin
exports.getUsers = asyncHandler(async (req, res, next) => {
  res.status(200).json(res.advancedResults);
});

// @desc    Get single user
// @route   GET /api/v1/admin/users/:id
// @access  Private/Admin
exports.getUser = asyncHandler(async (req, res, next) => {
  const user = await User.findById(req.params.id);

  if (!user) {
    return next(new ErrorResponse(`User not found with id of ${req.params.id}`, 404));
  }

  res.status(200).json({
    success: true,
    data: user
  });
});

// @desc    Create user
// @route   POST /api/v1/admin/users
// @access  Private/Admin
exports.createUser = asyncHandler(async (req, res, next) => {
  const user = await User.create(req.body);

  res.status(201).json({
    success: true,
    data: user
  });
});

// @desc    Update user
// @route   PUT /api/v1/admin/users/:id
// @access  Private/Admin
exports.updateUser = asyncHandler(async (req, res, next) => {
  const user = await User.findByIdAndUpdate(req.params.id, req.body, {
    new: true,
    runValidators: true
  });

  if (!user) {
    return next(new ErrorResponse(`User not found with id of ${req.params.id}`, 404));
  }

  res.status(200).json({
    success: true,
    data: user
  });
});

// @desc    Delete user
// @route   DELETE /api/v1/admin/users/:id
// @access  Private/Admin
exports.deleteUser = asyncHandler(async (req, res, next) => {
  const user = await User.findById(req.params.id);

  if (!user) {
    return next(new ErrorResponse(`User not found with id of ${req.params.id}`, 404));
  }

  await user.remove();

  res.status(200).json({
    success: true,
    data: {}
  });
});

// @desc    Get admin dashboard stats
// @route   GET /api/v1/admin/dashboard
// @access  Private/Admin
exports.getDashboardStats = asyncHandler(async (req, res, next) => {
  // Get user stats
  const totalUsers = await User.countDocuments();
  const premiumUsers = await User.countDocuments({ role: 'premium' });
  const regularUsers = await User.countDocuments({ role: 'user' });
  
  // Get recent users
  const recentUsers = await User.find()
    .sort({ registrationDate: -1 })
    .limit(5)
    .select('username email registrationDate');
  
  // Get payment stats
  const pendingVerifications = await PaymentVerification.countDocuments({ status: 'pending' });
  const approvedVerifications = await PaymentVerification.countDocuments({ status: 'approved' });
  const rejectedVerifications = await PaymentVerification.countDocuments({ status: 'rejected' });
  
  // Get recent verifications
  const recentVerifications = await PaymentVerification.find()
    .sort({ submissionDate: -1 })
    .limit(5)
    .populate({
      path: 'userId',
      select: 'username email'
    });
  
  // Get content stats
  const totalContent = await Content.countDocuments();
  const moduleCount = await Content.countDocuments({ type: 'module' });
  const lessonCount = await Content.countDocuments({ type: 'lesson' });
  
  res.status(200).json({
    success: true,
    data: {
      users: {
        total: totalUsers,
        premium: premiumUsers,
        regular: regularUsers,
        recent: recentUsers
      },
      payments: {
        pending: pendingVerifications,
        approved: approvedVerifications,
        rejected: rejectedVerifications,
        recent: recentVerifications
      },
      content: {
        total: totalContent,
        modules: moduleCount,
        lessons: lessonCount
      }
    }
  });
});

// @desc    Create admin user
// @route   POST /api/v1/admin/setup
// @access  Public (one-time setup)
exports.createAdminUser = asyncHandler(async (req, res, next) => {
  // Check if admin already exists
  const adminExists = await User.findOne({ role: 'admin' });
  
  if (adminExists) {
    return next(new ErrorResponse('Admin user already exists', 400));
  }
  
  const { username, email, password } = req.body;
  
  // Create admin user
  const admin = await User.create({
    username,
    email,
    password,
    role: 'admin'
  });
  
  res.status(201).json({
    success: true,
    message: 'Admin user created successfully',
    data: {
      id: admin._id,
      username: admin.username,
      email: admin.email,
      role: admin.role
    }
  });
});
