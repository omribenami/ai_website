# AI Expert Guide Blog Structure

## Site Architecture

```
AI Expert Blog
├── Public Pages
│   ├── Home
│   ├── About
│   ├── Blog Preview
│   ├── Login
│   └── Register
├── Free Content (Registered Users)
│   ├── Module 1: Introduction to AI
│   │   ├── Lesson 1.1: What is AI?
│   │   ├── Lesson 1.2: History of AI
│   │   ├── Lesson 1.3: Types of AI
│   │   └── Lesson 1.4: AI Ethics
│   └── Community Forum (Preview)
├── Premium Content (Paid Users)
│   ├── Module 2: Machine Learning Fundamentals
│   │   ├── Lesson 2.1: Introduction to ML
│   │   ├── Lesson 2.2: Supervised Learning
│   │   ├── Lesson 2.3: Unsupervised Learning
│   │   └── Lesson 2.4: Practical ML Projects
│   ├── Module 3: Deep Learning
│   │   ├── Lesson 3.1: Neural Networks Basics
│   │   ├── Lesson 3.2: CNN Architecture
│   │   ├── Lesson 3.3: RNN and LSTM
│   │   └── Lesson 3.4: Transformers
│   ├── Module 4: AI Development Environment
│   │   ├── Lesson 4.1: Setting Up Your Environment
│   │   ├── Lesson 4.2: Essential Python Libraries
│   │   ├── Lesson 4.3: GPU Configuration
│   │   └── Lesson 4.4: Cloud AI Services
│   ├── Interactive Exercises
│   │   ├── Code Challenges
│   │   ├── Quizzes
│   │   └── Projects
│   └── Community Forum (Full Access)
└── Admin Area
    ├── Dashboard
    ├── User Management
    ├── Content Management
    ├── Payment Verification
    └── Analytics
```

## Database Schema

### Users Collection
```json
{
  "_id": "ObjectId",
  "username": "String",
  "email": "String",
  "passwordHash": "String",
  "role": "String (user, premium, admin)",
  "registrationDate": "Date",
  "lastLogin": "Date",
  "profile": {
    "name": "String",
    "bio": "String",
    "avatar": "String (URL)"
  },
  "progress": {
    "completedLessons": ["LessonId"],
    "quizScores": [
      {
        "quizId": "String",
        "score": "Number",
        "completedDate": "Date"
      }
    ]
  },
  "payment": {
    "status": "String (none, pending, verified)",
    "verificationCode": "String",
    "paymentDate": "Date",
    "verificationDate": "Date"
  }
}
```

### Content Collection
```json
{
  "_id": "ObjectId",
  "title": "String",
  "slug": "String",
  "type": "String (module, lesson, quiz, exercise)",
  "content": "String (Markdown)",
  "accessLevel": "String (free, premium)",
  "parentId": "ObjectId (for lessons within modules)",
  "order": "Number",
  "createdDate": "Date",
  "updatedDate": "Date",
  "metadata": {
    "readTime": "Number",
    "difficulty": "String (beginner, intermediate, advanced)",
    "tags": ["String"]
  },
  "resources": [
    {
      "type": "String (video, code, download)",
      "url": "String",
      "title": "String"
    }
  ]
}
```

### Comments Collection
```json
{
  "_id": "ObjectId",
  "contentId": "ObjectId",
  "userId": "ObjectId",
  "text": "String",
  "createdDate": "Date",
  "updatedDate": "Date",
  "parentCommentId": "ObjectId (for replies)",
  "likes": ["UserId"]
}
```

### PaymentVerification Collection
```json
{
  "_id": "ObjectId",
  "userId": "ObjectId",
  "submissionDate": "Date",
  "verificationCode": "String",
  "status": "String (pending, approved, rejected)",
  "reviewedBy": "ObjectId (adminId)",
  "reviewDate": "Date",
  "notes": "String",
  "evidence": "String (URL to screenshot/receipt)"
}
```

## Page Designs

### Home Page
- Hero section with course introduction
- Featured content preview
- Testimonials/success stories
- Clear CTA for registration
- Preview of Module 1 content
- Pricing information ($10 for full access)

### User Dashboard
- Progress tracking
- Recommended next lessons
- Bookmarked content
- Recent activity
- Access level indicator
- Upgrade prompt (for free users)

### Lesson Page
- Content area with Markdown rendering
- Code syntax highlighting
- Interactive elements (where applicable)
- Progress tracking
- Navigation between lessons
- Comments/discussion section
- Related lessons
- Premium content indicators/teasers

### Payment Verification Page
- Instructions for making payment
- Buy Me a Coffee link
- Verification code entry
- Receipt upload option
- Status indicator
- FAQ about the process

### Admin Dashboard
- User statistics
- Content engagement metrics
- Pending verification requests
- Recent registrations
- Quick access to management tools

## User Interface Components

### Navigation
- Responsive navbar
- Sidebar for lesson navigation
- Breadcrumbs for deep content
- Progress indicators
- Access level indicators

### Content Display
- Markdown renderer
- Code editor/viewer
- Interactive visualizations
- Quiz interface
- Video embeds

### User Authentication
- Login/registration forms
- Password reset flow
- Profile editor
- Access level badges

### Payment Process
- Payment instructions
- Verification submission form
- Status indicators
- Admin verification interface

## Access Control Logic

### Public Access
- Home page
- About page
- Registration
- Login
- Blog previews/teasers

### Free User Access
- Module 1 (complete)
- Preview of premium content
- Limited community features
- Profile management

### Premium User Access
- All modules
- Interactive exercises
- Full community access
- Downloadable resources

### Admin Access
- All user and premium content
- User management
- Content management
- Payment verification
- Analytics dashboard

## Responsive Design Considerations

- Mobile-first approach
- Collapsible navigation for small screens
- Readable typography at all sizes
- Touch-friendly interactive elements
- Optimized images and media
- Simplified layouts for mobile views

## Technical Implementation Notes

- Use React Router for navigation
- JWT for authentication
- Markdown parsing with react-markdown
- Code syntax highlighting with react-syntax-highlighter
- Form validation with Formik or React Hook Form
- State management with Context API or Redux
- Material UI for consistent design components
- Responsive design with CSS Grid and Flexbox
- MongoDB for flexible content and user data storage
