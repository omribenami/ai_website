import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Box, Typography, Paper } from '@mui/material';

const MarkdownRenderer = ({ content }) => {
  return (
    <Box component={Paper} sx={{ p: 3, my: 2 }}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          h1: ({ node, ...props }) => (
            <Typography variant="h1" gutterBottom {...props} />
          ),
          h2: ({ node, ...props }) => (
            <Typography variant="h2" gutterBottom {...props} />
          ),
          h3: ({ node, ...props }) => (
            <Typography variant="h3" gutterBottom {...props} />
          ),
          h4: ({ node, ...props }) => (
            <Typography variant="h4" gutterBottom {...props} />
          ),
          h5: ({ node, ...props }) => (
            <Typography variant="h5" gutterBottom {...props} />
          ),
          h6: ({ node, ...props }) => (
            <Typography variant="h6" gutterBottom {...props} />
          ),
          p: ({ node, ...props }) => (
            <Typography variant="body1" paragraph {...props} />
          ),
          a: ({ node, ...props }) => (
            <Typography
              component="a"
              sx={{ color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
              {...props}
            />
          ),
          code: ({ node, inline, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={tomorrow}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <Typography
                component="code"
                sx={{
                  backgroundColor: 'grey.100',
                  padding: '2px 4px',
                  borderRadius: 1,
                  fontFamily: 'Fira Code, monospace',
                }}
                {...props}
              >
                {children}
              </Typography>
            );
          },
          ul: ({ node, ...props }) => (
            <Typography component="ul" sx={{ pl: 4, mb: 2 }} {...props} />
          ),
          ol: ({ node, ...props }) => (
            <Typography component="ol" sx={{ pl: 4, mb: 2 }} {...props} />
          ),
          li: ({ node, ...props }) => (
            <Typography component="li" sx={{ mb: 1 }} {...props} />
          ),
          blockquote: ({ node, ...props }) => (
            <Box
              component="blockquote"
              sx={{
                borderLeft: 4,
                borderColor: 'primary.main',
                pl: 2,
                py: 1,
                my: 2,
                bgcolor: 'grey.50',
              }}
              {...props}
            />
          ),
          img: ({ node, ...props }) => (
            <Box
              component="img"
              sx={{
                maxWidth: '100%',
                height: 'auto',
                display: 'block',
                my: 2,
                borderRadius: 1,
              }}
              {...props}
            />
          ),
          table: ({ node, ...props }) => (
            <Box
              component="table"
              sx={{
                width: '100%',
                borderCollapse: 'collapse',
                my: 2,
              }}
              {...props}
            />
          ),
          th: ({ node, ...props }) => (
            <Box
              component="th"
              sx={{
                borderBottom: 1,
                borderColor: 'grey.300',
                p: 1,
                textAlign: 'left',
                fontWeight: 'bold',
              }}
              {...props}
            />
          ),
          td: ({ node, ...props }) => (
            <Box
              component="td"
              sx={{
                borderBottom: 1,
                borderColor: 'grey.300',
                p: 1,
              }}
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </Box>
  );
};

export default MarkdownRenderer;
