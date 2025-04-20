import React from 'react';
import {
    Box, Checkbox,
    FormControl, FormControlLabel, FormGroup,
    MenuItem,
    Select,
    Slider,
    ToggleButton,
    ToggleButtonGroup,
    Tooltip,
    Typography
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import {Language, QuestionType} from '../types';

interface QuestionComponentProps {
    currentQuestion: QuestionType;
    language: Language;
    answers: Record<string, any>;
    handleAnswer: (id: string, value: any) => void;
}

const QuestionComponent: React.FC<QuestionComponentProps> = ({
                                                                 currentQuestion,
                                                                 language,
                                                                 answers,
                                                                 handleAnswer
                                                             }) => {
    const currentAnswer = answers[currentQuestion.name] ?? currentQuestion.default_value;

    const renderInput = () => {
        switch (currentQuestion.data_type) {
            case 'B':
                return (
                    <ToggleButtonGroup
                        color="primary"
                        exclusive
                        value={currentAnswer}
                        onChange={(_, value) => handleAnswer(currentQuestion.name, value)}
                        fullWidth
                    >
                        <ToggleButton
                            value={true}
                            sx={{
                                fontWeight: currentAnswer === true ? 'bold' : 'normal',
                                bgcolor: currentAnswer === true ? 'rgba(25, 118, 210, 0.12)' : 'inherit'
                            }}
                        >
                            {language === 'en' ? 'Yes' : language === 'fr' ? 'Oui' : 'Да'}
                        </ToggleButton>
                        <ToggleButton
                            value={false}
                            sx={{
                                fontWeight: currentAnswer === false ? 'bold' : 'normal',
                                bgcolor: currentAnswer === false ? 'rgba(25, 118, 210, 0.12)' : 'inherit'
                            }}
                        >
                            {language === 'en' ? 'No' : language === 'fr' ? 'Non' : 'Нет'}
                        </ToggleButton>
                    </ToggleButtonGroup>
                );

            case 'C':
                return (
                    <Select
                        value={currentAnswer || currentQuestion.default_value}
                        onChange={(e) => handleAnswer(currentQuestion.name, e.target.value)}
                        fullWidth
                        sx={{
                            '& .MuiSelect-select': { fontWeight: 600, color: 'primary.main' }
                        }}
                    >
                        {currentQuestion['possible-values'].map((value) => (
                            <MenuItem
                                key={value.toString()}
                                value={value}
                                sx={{
                                    fontWeight: value === currentAnswer ? 'bold' : 'normal',
                                    bgcolor: value === currentAnswer ? 'action.selected' : 'inherit'
                                }}
                            >
                                {currentQuestion.value_meaning[value.toString()]?.[language] || value}
                            </MenuItem>
                        ))}
                    </Select>
                );

            case 'R':
                const numericValues = currentQuestion['possible-values']
                    .map(val => Number(val))
                    .filter(val => !Number.isNaN(val));
                if (numericValues.length === 0) return <Typography color="error">Invalid values</Typography>;

                return (
                    <Slider
                        value={currentAnswer || Number(currentQuestion.default_value)}
                        min={Math.min(...numericValues)}
                        max={Math.max(...numericValues)}
                        step={1}
                        marks={numericValues.map(value => ({
                            value,
                            label: currentQuestion.value_meaning[value]?.[language] || value
                        }))}
                        valueLabelDisplay="auto"
                        onChange={(_, value) => handleAnswer(currentQuestion.name, value)}
                        sx={{
                            '& .MuiSlider-valueLabel': { backgroundColor: 'primary.main', fontWeight: 'bold' }
                        }}
                    />
                );

            case 'M':
                return (
                    <FormGroup>
                        {currentQuestion['possible-values'].map((value) => (
                            <FormControlLabel
                                key={value.toString()}
                                control={
                                    <Checkbox
                                        checked={currentAnswer.includes(value)}
                                        onChange={(e) => {
                                            const newValue = e.target.checked
                                                ? [...currentAnswer, value]
                                                : currentAnswer.filter((v: string | number) => v !== value); // Явный тип
                                            handleAnswer(currentQuestion.name, newValue);
                                        }}
                                    />
                                }
                                label={currentQuestion.value_meaning[value]?.[language] || value}
                            />
                        ))}
                    </FormGroup>
                );

            default:
                return <Typography color="error">Unsupported type</Typography>;
        }
    };

    return (
        <FormControl component="fieldset" fullWidth sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 'bold', mr: 1 }}>
                    {currentQuestion[`question_${language}`]}
                </Typography>
                {currentQuestion.description_ru && (
                    <Tooltip title={currentQuestion[`description_${language}`]} arrow>
                        <HelpOutlineIcon sx={{ fontSize: '1rem', color: 'text.secondary' }} />
                    </Tooltip>
                )}
            </Box>
            {renderInput()}
        </FormControl>
    );
};

export default QuestionComponent;