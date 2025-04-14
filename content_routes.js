// Content Routes for AI Expert Guide Blog

// server/routes/content.js
const express = require('express');
const {
  getAllContent,
  getContent,
  getContentBySlug,
  getModuleWithLessons,
  createContent,
  updateContent,
  deleteContent,
  trackProgress
} = require('../controllers/content');

const router = express.Router();

// Import middleware
const { protect, authorize } = require('../middleware/auth');

// Public routes (some with conditional access based on content type)
router.get('/', getAllContent);
router.get('/slug/:slug', getContentBySlug);
router.get('/:id', getContent);
router.get('/module/:moduleId', getModuleWithLessons);

// Protected routes
router.post('/:id/progress', protect, trackProgress);

// Admin only routes
router.post('/', protect, authorize('admin'), createContent);
router.put('/:id', protect, authorize('admin'), updateContent);
router.delete('/:id', protect, authorize('admin'), deleteContent);

module.exports = router;
