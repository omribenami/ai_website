// Admin Routes for AI Expert Guide Blog

// server/routes/admin.js
const express = require('express');
const {
  getUsers,
  getUser,
  createUser,
  updateUser,
  deleteUser,
  getDashboardStats,
  createAdminUser
} = require('../controllers/admin');

const User = require('../models/User');
const advancedResults = require('../middleware/advancedResults');

const router = express.Router();

// Import middleware
const { protect, authorize } = require('../middleware/auth');

// Apply protection to all routes except admin setup
router.use(protect);
router.use(authorize('admin'));

// Routes
router.get('/dashboard', getDashboardStats);
router.get('/users', advancedResults(User), getUsers);
router.get('/users/:id', getUser);
router.post('/users', createUser);
router.put('/users/:id', updateUser);
router.delete('/users/:id', deleteUser);

// Public route for initial admin setup (no protection)
router.post('/setup', createAdminUser);

module.exports = router;
