from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "sagorsarker/mbert-bengali-tydiqa-qa"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
qa_input = {
    'question': 'মাস্টারদা সূর্যকুমার সেনের বাবার নাম কী ছিল ?',
    'context': 'সূর্য সেন ১৮৯৪ সালের ২২ মার্চ চট্টগ্রামের রাউজান থানার নোয়াপাড়ায় অর্থনৈতিক ভাবে অস্বচ্ছল পরিবারে জন্মগ্রহণ করেন। তাঁর পিতার নাম রাজমনি সেন এবং মাতার নাম শশী বালা সেন। রাজমনি সেনের দুই ছেলে আর চার মেয়ে। সূর্য সেন তাঁদের পরিবারের চতুর্থ সন্তান। দুই ছেলের নাম সূর্য ও কমল। চার মেয়ের নাম বরদাসুন্দরী, সাবিত্রী, ভানুমতী ও প্রমিলা। শৈশবে পিতা মাতাকে হারানো সূর্য সেন কাকা গৌরমনি সেনের কাছে মানুষ হয়েছেন। সূর্য সেন ছেলেবেলা থেকেই খুব মনোযোগী ভাল ছাত্র ছিলেন এবং ধর্মভাবাপন্ন গম্ভীর প্রকৃতির ছিলেন।'
}
result = nlp(qa_input)
print(result)
