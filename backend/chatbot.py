class FarmerChatbot:
    """Simple template-based chatbot for a demo-friendly farming assistant."""

    def __init__(self) -> None:
        self.default_response = (
            "I can help with crop disease, irrigation, fertilizer, pest control, "
            "and organic farming tips. Try asking a short farming question."
        )

    def get_response(self, message: str) -> str:
        text = message.lower().strip()

        if any(word in text for word in ["hello", "hi", "hey", "namaste"]):
            return (
                "Hello farmer! You can ask me about plant disease, watering schedules, "
                "soil health, pests, or fertilizer guidance."
            )

        if any(word in text for word in ["disease", "leaf spot", "blight", "rust", "yellow leaves"]):
            return (
                "Start by removing badly affected leaves, keep the crop area dry and clean, "
                "and monitor how quickly the symptoms spread. You can also upload a plant image "
                "above for a quick disease prediction."
            )

        if any(word in text for word in ["water", "irrigation", "drip", "moisture"]):
            return (
                "Water early in the morning and avoid over-irrigation. Check soil moisture first, "
                "because roots need oxygen as much as they need water."
            )

        if any(word in text for word in ["fertilizer", "nutrient", "urea", "npk", "compost"]):
            return (
                "Use balanced nutrition based on crop stage. Compost improves soil structure, "
                "while chemical fertilizer should be applied in recommended doses to avoid root stress."
            )

        if any(word in text for word in ["pest", "insect", "aphid", "worm", "caterpillar"]):
            return (
                "Inspect the underside of leaves, remove badly infested parts, and consider neem-based "
                "sprays first. If the infestation is severe, use a crop-specific pesticide carefully."
            )

        if any(word in text for word in ["organic", "natural", "bio"]):
            return (
                "Organic support options include neem oil spray, compost tea, crop rotation, "
                "mulching, and improving airflow between plants."
            )

        if any(word in text for word in ["thank", "thanks"]):
            return "You're welcome. Keep the questions coming, and we’ll make this demo useful for farmers."

        return self.default_response
