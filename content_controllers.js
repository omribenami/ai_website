// Content Controllers for AI Expert Guide Blog

// server/controllers/content.js
const Content = require('../models/Content');
const ErrorResponse = require('../utils/errorResponse');
const asyncHandler = require('../middleware/async');
const slugify = require('slugify');

// @desc    Get all content
// @route   GET /api/v1/content
// @access  Public/Private (filtered by access level)
exports.getAllContent = asyncHandler(async (req, res, next) => {
  // For non-authenticated users, only return public info about content
  if (!req.user) {
    const publicContent = await Content.find({ type: 'module' })
      .select('title slug type accessLevel metadata.difficulty metadata.readTime')
      .sort('order');
    
    return res.status(200).json({
      success: true,
      count: publicContent.length,
      data: publicContent
    });
  }
  
  // For authenticated users, filter based on their role
  let query = {};
  
  // Regular users can only see free content
  if (req.user.role === 'user') {
    query.accessLevel = 'free';
  }
  
  // Admin and premium users can see all content
  const content = await Content.find(query).sort('order');
  
  res.status(200).json({
    success: true,
    count: content.length,
    data: content
  });
});

// @desc    Get single content
// @route   GET /api/v1/content/:id
// @access  Public/Private (filtered by access level)
exports.getContent = asyncHandler(async (req, res, next) => {
  const content = await Content.findById(req.params.id);
  
  if (!content) {
    return next(new ErrorResponse(`Content not found with id of ${req.params.id}`, 404));
  }
  
  // Check if content is premium and user has access
  if (content.accessLevel === 'premium') {
    // If no user is logged in
    if (!req.user) {
      return next(new ErrorResponse('Not authorized to access this content', 401));
    }
    
    // If user is not premium or admin
    if (req.user.role !== 'premium' && req.user.role !== 'admin') {
      return next(new ErrorResponse('Premium content requires upgrade', 403));
    }
  }
  
  res.status(200).json({
    success: true,
    data: content
  });
});

// @desc    Get content by slug
// @route   GET /api/v1/content/slug/:slug
// @access  Public/Private (filtered by access level)
exports.getContentBySlug = asyncHandler(async (req, res, next) => {
  const content = await Content.findOne({ slug: req.params.slug });
  
  if (!content) {
    return next(new ErrorResponse(`Content not found with slug of ${req.params.slug}`, 404));
  }
  
  // Check if content is premium and user has access
  if (content.accessLevel === 'premium') {
    // If no user is logged in
    if (!req.user) {
      return next(new ErrorResponse('Not authorized to access this content', 401));
    }
    
    // If user is not premium or admin
    if (req.user.role !== 'premium' && req.user.role !== 'admin') {
      return next(new ErrorResponse('Premium content requires upgrade', 403));
    }
  }
  
  res.status(200).json({
    success: true,
    data: content
  });
});

// @desc    Get module with lessons
// @route   GET /api/v1/content/module/:moduleId
// @access  Public/Private (filtered by access level)
exports.getModuleWithLessons = asyncHandler(async (req, res, next) => {
  const module = await Content.findOne({ 
    _id: req.params.moduleId,
    type: 'module'
  });
  
  if (!module) {
    return next(new ErrorResponse(`Module not found with id of ${req.params.moduleId}`, 404));
  }
  
  // Check if module is premium and user has access
  if (module.accessLevel === 'premium') {
    // If no user is logged in
    if (!req.user) {
      return next(new ErrorResponse('Not authorized to access this content', 401));
    }
    
    // If user is not premium or admin
    if (req.user.role !== 'premium' && req.user.role !== 'admin') {
      return next(new ErrorResponse('Premium content requires upgrade', 403));
    }
  }
  
  // Get lessons for this module
  let query = { 
    parentId: module._id,
    type: 'lesson'
  };
  
  // Regular users can only see free lessons
  if (req.user && req.user.role === 'user') {
    query.accessLevel = 'free';
  }
  
  const lessons = await Content.find(query).sort('order');
  
  res.status(200).json({
    success: true,
    data: {
      module,
      lessons
    }
  });
});

// @desc    Create content
// @route   POST /api/v1/content
// @access  Private/Admin
exports.createContent = asyncHandler(async (req, res, next) => {
  // Create slug from title
  if (req.body.title && !req.body.slug) {
    req.body.slug = slugify(req.body.title, { lower: true });
  }
  
  const content = await Content.create(req.body);
  
  res.status(201).json({
    success: true,
    data: content
  });
});

// @desc    Update content
// @route   PUT /api/v1/content/:id
// @access  Private/Admin
exports.updateContent = asyncHandler(async (req, res, next) => {
  // Update slug if title changes
  if (req.body.title && !req.body.slug) {
    req.body.slug = slugify(req.body.title, { lower: true });
  }
  
  const content = await Content.findByIdAndUpdate(req.params.id, req.body, {
    new: true,
    runValidators: true
  });
  
  if (!content) {
    return next(new ErrorResponse(`Content not found with id of ${req.params.id}`, 404));
  }
  
  res.status(200).json({
    success: true,
    data: content
  });
});

// @desc    Delete content
// @route   DELETE /api/v1/content/:id
// @access  Private/Admin
exports.deleteContent = asyncHandler(async (req, res, next) => {
  const content = await Content.findById(req.params.id);
  
  if (!content) {
    return next(new ErrorResponse(`Content not found with id of ${req.params.id}`, 404));
  }
  
  // If deleting a module, also delete all its lessons
  if (content.type === 'module') {
    await Content.deleteMany({ parentId: content._id });
  }
  
  await content.remove();
  
  res.status(200).json({
    success: true,
    data: {}
  });
});

// @desc    Track user progress
// @route   POST /api/v1/content/:id/progress
// @access  Private
exports.trackProgress = asyncHandler(async (req, res, next) => {
  const content = await Content.findById(req.params.id);
  
  if (!content) {
    return next(new ErrorResponse(`Content not found with id of ${req.params.id}`, 404));
  }
  
  // Check if content is premium and user has access
  if (content.accessLevel === 'premium') {
    if (req.user.role !== 'premium' && req.user.role !== 'admin') {
      return next(new ErrorResponse('Premium content requires upgrade', 403));
    }
  }
  
  // Add to user's completed lessons if not already there
  if (!req.user.progress.completedLessons.includes(req.params.id)) {
    req.user.progress.completedLessons.push(req.params.id);
    await req.user.save();
  }
  
  res.status(200).json({
    success: true,
    data: req.user.progress
  });
});
