import React from 'react';
import { Alert, Box, Container, LinearProgress, Paper, Typography } from '@mui/material';
import { useLocation, useNavigate } from 'react-router-dom';
import { Prediction } from './types';
import { Button } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';

const ResultsPage: React.FC = () => {
    const { state } = useLocation();
    const predictions: Prediction[] = state?.predictions || [];
    const navigate = useNavigate();

    const sortedPredictions = [...predictions].sort((a, b) => b.probability - a.probability);
    const topPredictions = sortedPredictions.slice(0, 5);

    if (!predictions.length) {
        return (
            <Container maxWidth="md" sx={{ mt: 4 }}>
                <Alert severity="error">No results found. Please complete the survey again.</Alert>
                <Button
                    variant="contained"
                    sx={{ mt: 2 }}
                    onClick={() => navigate('/')}
                >
                    Return to Home
                </Button>
            </Container>
        );
    }

    return (
        <Container maxWidth="md" sx={{ py: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
                Diagnostic Results
            </Typography>

            <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 4 }}>
                {topPredictions.map((prediction, index) => (
                    <Box key={prediction.disease} sx={{ mb: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="h6">
                                {index + 1}. {prediction.disease}
                            </Typography>
                            <Typography variant="body1">
                                {(prediction.probability * 100).toFixed(2)}%
                            </Typography>
                        </Box>
                        <LinearProgress
                            variant="determinate"
                            value={prediction.probability * 100}
                            sx={{ height: 10, borderRadius: 5 }}
                        />
                    </Box>
                ))}
            </Paper>

            <Button
                variant="contained"
                startIcon={<HomeIcon />}
                onClick={() => navigate('/')}
            >
                Return to Home
            </Button>
        </Container>
    );
};

export default ResultsPage;