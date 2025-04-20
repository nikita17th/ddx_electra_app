export type Language = 'en' | 'fr' | 'ru';

export type QuestionType = {
    name: string;
    code_question: string;
    question_en: string;
    question_fr: string;
    question_ru: string;
    is_antecedent: boolean;
    default_value: string | number | boolean | [];
    value_meaning: {
        [key: string]: {
            en: string;
            fr: string;
            ru: string;
        };
    };
    "possible-values": (string | number)[];
    data_type: 'B' | 'M' | 'C' | 'R';
    description_en?: string;
    description_fr?: string;
    description_ru?: string;
}

export type Prediction = {
    disease: string;
    probability: number;
}