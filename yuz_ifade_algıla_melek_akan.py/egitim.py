from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Veriyi oku (Türkçe karakterlere uygun encoding)
df = pd.read_csv("veriseti.csv", encoding="ISO-8859-9")

# Etiket ve özellikleri ayır
y = df["Etiket"]
X = df.drop("Etiket", axis=1)

# %80 eğitim - %20 test olarak ayır
Xegt, Xtst, Yegt, Ytst = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test verilerini ekrana yazdır
print("📊 Eğitim verisi (X %80):")
print(Xegt)
print("\n🧪 Test verisi (X %20):")
print(Xtst)
print("\n🎯 Eğitim etiketleri:")
print(Yegt)
print("\n🔍 Test etiketleri:")
print(Ytst)

# Pipeline: normalizasyon + rastgele orman sınıflandırıcı
pipeline = Pipeline([
    ("std", StandardScaler()),
    ("sinif", RandomForestClassifier(random_state=42))
])

# Modeli eğit
pipeline.fit(Xegt, Yegt)

# Test verisinde tahmin yap
Y_model = pipeline.predict(Xtst)

# Doğruluk oranını yazdır
dogruluk_orani = accuracy_score(Ytst, Y_model)
print(f"\n✅ Doğruluk Oranı: {dogruluk_orani}")

# Eğitilen modeli dosyaya kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
