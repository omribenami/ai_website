import React, { useState } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  IconButton, 
  Drawer, 
  List, 
  ListItem, 
  ListItemText, 
  Box, 
  Avatar, 
  Menu, 
  MenuItem, 
  Divider,
  Badge,
  useMediaQuery,
  useTheme
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import { Link as RouterLink } from 'react-router-dom';

const ResponsiveNavbar = ({ isAuthenticated = false }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationsAnchor, setNotificationsAnchor] = useState(null);
  
  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleNotificationsOpen = (event) => {
    setNotificationsAnchor(event.currentTarget);
  };
  
  const handleNotificationsClose = () => {
    setNotificationsAnchor(null);
  };
  
  const menuItems = [
    { text: 'Home', path: '/' },
    { text: 'Courses', path: '/courses' },
    { text: 'About', path: '/about' },
    { text: 'Contact', path: '/contact' }
  ];
  
  const authenticatedMenuItems = [
    { text: 'Dashboard', path: '/dashboard' },
    { text: 'My Courses', path: '/my-courses' },
    { text: 'Profile', path: '/profile' },
    { text: 'Logout', path: '/logout' }
  ];
  
  const notifications = [
    { id: 1, text: 'New course available: Advanced Deep Learning', read: false },
    { id: 2, text: 'Your quiz result: Machine Learning Fundamentals - 85%', read: true },
    { id: 3, text: 'Course update: Python for AI has new content', read: false }
  ];
  
  const unreadNotifications = notifications.filter(notification => !notification.read).length;
  
  const drawer = (
    <Box sx={{ width: 250 }} role="presentation">
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
        <IconButton onClick={handleDrawerToggle}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem 
            button 
            key={item.text} 
            component={RouterLink} 
            to={item.path}
            onClick={handleDrawerToggle}
          >
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
      <Divider />
      {isAuthenticated ? (
        <List>
          {authenticatedMenuItems.map((item) => (
            <ListItem 
              button 
              key={item.text} 
              component={RouterLink} 
              to={item.path}
              onClick={handleDrawerToggle}
            >
              <ListItemText primary={item.text} />
            </ListItem>
          ))}
        </List>
      ) : (
        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Button 
            variant="contained" 
            color="primary" 
            fullWidth
            component={RouterLink}
            to="/login"
            onClick={handleDrawerToggle}
          >
            Login
          </Button>
          <Button 
            variant="outlined" 
            color="primary" 
            fullWidth
            component={RouterLink}
            to="/register"
            onClick={handleDrawerToggle}
          >
            Register
          </Button>
        </Box>
      )}
    </Box>
  );
  
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Typography 
            variant="h6" 
            component={RouterLink} 
            to="/" 
            sx={{ 
              flexGrow: 1, 
              textDecoration: 'none', 
              color: 'inherit',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center'
            }}
          >
            AI Expert Guide
          </Typography>
          
          {!isMobile && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {menuItems.map((item) => (
                <Button 
                  key={item.text} 
                  color="inherit" 
                  component={RouterLink} 
                  to={item.path}
                  sx={{ mx: 1 }}
                >
                  {item.text}
                </Button>
              ))}
            </Box>
          )}
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <IconButton color="inherit" component={RouterLink} to="/search">
              <SearchIcon />
            </IconButton>
            
            {isAuthenticated && (
              <>
                <IconButton 
                  color="inherit" 
                  onClick={handleNotificationsOpen}
                  aria-controls="notifications-menu"
                  aria-haspopup="true"
                >
                  <Badge badgeContent={unreadNotifications} color="error">
                    <NotificationsIcon />
                  </Badge>
                </IconButton>
                
                <Menu
                  id="notifications-menu"
                  anchorEl={notificationsAnchor}
                  keepMounted
                  open={Boolean(notificationsAnchor)}
                  onClose={handleNotificationsClose}
                  PaperProps={{
                    style: {
                      width: '320px',
                      maxHeight: '400px',
                    },
                  }}
                >
                  <Box sx={{ p: 2 }}>
                    <Typography variant="h6">Notifications</Typography>
                  </Box>
                  <Divider />
                  {notifications.length > 0 ? (
                    notifications.map((notification) => (
                      <MenuItem 
                        key={notification.id} 
                        onClick={handleNotificationsClose}
                        sx={{ 
                          whiteSpace: 'normal',
                          bgcolor: notification.read ? 'transparent' : 'action.hover'
                        }}
                      >
                        <Typography variant="body2">{notification.text}</Typography>
                      </MenuItem>
                    ))
                  ) : (
                    <MenuItem>
                      <Typography variant="body2">No notifications</Typography>
                    </MenuItem>
                  )}
                  <Divider />
                  <MenuItem 
                    component={RouterLink} 
                    to="/notifications"
                    onClick={handleNotificationsClose}
                    sx={{ justifyContent: 'center' }}
                  >
                    <Typography variant="body2" color="primary">View all notifications</Typography>
                  </MenuItem>
                </Menu>
                
                <IconButton color="inherit" component={RouterLink} to="/checkout">
                  <ShoppingCartIcon />
                </IconButton>
              </>
            )}
            
            {isAuthenticated ? (
              <>
                <IconButton
                  edge="end"
                  aria-label="account of current user"
                  aria-controls="menu-appbar"
                  aria-haspopup="true"
                  onClick={handleProfileMenuOpen}
                  color="inherit"
                >
                  <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>U</Avatar>
                </IconButton>
                <Menu
                  id="menu-appbar"
                  anchorEl={anchorEl}
                  anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'right',
                  }}
                  keepMounted
                  transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                  }}
                  open={Boolean(anchorEl)}
                  onClose={handleProfileMenuClose}
                >
                  {authenticatedMenuItems.map((item) => (
                    <MenuItem 
                      key={item.text} 
                      component={RouterLink} 
                      to={item.path}
                      onClick={handleProfileMenuClose}
                    >
                      {item.text}
                    </MenuItem>
                  ))}
                </Menu>
              </>
            ) : (
              !isMobile && (
                <Box sx={{ display: 'flex', ml: 2 }}>
                  <Button 
                    color="inherit" 
                    component={RouterLink} 
                    to="/login"
                    sx={{ mr: 1 }}
                  >
                    Login
                  </Button>
                  <Button 
                    variant="contained" 
                    color="primary"
                    component={RouterLink}
                    to="/register"
                  >
                    Register
                  </Button>
                </Box>
              )
            )}
          </Box>
        </Toolbar>
      </AppBar>
      
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={handleDrawerToggle}
      >
        {drawer}
      </Drawer>
    </Box>
  );
};

export default ResponsiveNavbar;
