from pydantic import BaseModel, validator


class Entry(BaseModel):
    age: float = 67
    ever_married: bool = True
    Residence_type: str = "Urban"
    work_type: str = "Private"
    hypertension: bool = False
    bmi: float = 36.6
    avg_glucose_level: float = 228.69
    heart_disease: bool = True
    smoking_status: str = "formerly smoked"

    @property
    def neg_effect_on_lungs(self):
        if (self.smoking_status in ["smokes", "formerly smoked"]) and (
            self.Residence_type == "Urban"
        ):
            return "Yes"
        return "No"

    @property
    def heart_problems(self):
        if self.heart_disease or self.hypertension:
            return "Yes"
        return "No"

    @validator("age")
    def age_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError(f"we expect age >= 0, we received {value}")
        return value

    @validator("avg_glucose_level")
    def avg_glucose_level_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError(
                f"we expect avg_glucose_level >= 0, we received {value}")
        return value

    @validator("Residence_type")
    def Residence_type_must_be_of_type(cls, value):
        expected = ["urban", "rural"]
        if value.lower() not in expected:
            raise ValueError(
                f"we expect Residence_type to be {expected}, we received '{value}'"
            )
        return value

    @validator("work_type")
    def work_type_must_be_of_type(cls, value):
        expected = [
            "private",
            "never_worked",
            "govt_job",
            "children",
            "self-employed"]
        if value.lower() not in expected:
            raise ValueError(
                f"we expect work_type to be {expected}, we received '{value}'"
            )
        return value

    @validator("smoking_status")
    def smoking_status_must_be_of_type(cls, value):
        expected = ["unknown", "never smoked", "smokes", "formerly smoked"]
        if value.lower() not in expected:
            raise ValueError(
                f"we expect smoking_status to be {expected}, we received '{value}'"
            )
        return value

    def get_entry_dict(self):
        data = {}
        data["age"] = [self.age]
        data["hypertension"] = ["Yes"] if self.hypertension else ["No"]
        data["heart_disease"] = ["Yes"] if self.heart_disease else ["No"]
        data["ever_married"] = ["Yes"] if self.ever_married else ["No"]
        data["work_type"] = [self.work_type]
        data["avg_glucose_level"] = [self.avg_glucose_level]
        data["bmi"] = [self.bmi]
        data["smoking_status"] = [self.smoking_status]
        data["neg_effect_on_lungs"] = [self.neg_effect_on_lungs]
        data["heart_problems"] = ["Yes"] if self.heart_problems else ["No"]
        return data
