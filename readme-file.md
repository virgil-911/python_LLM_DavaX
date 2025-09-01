# Book Recommendation Chatbot with RAG

Un chatbot AI inteligent care recomandă cărți folosind OpenAI GPT, ChromaDB pentru RAG (Retrieval-Augmented Generation), și function calling pentru rezumate detaliate.

## Caracteristici Principale

- **RAG cu ChromaDB**: Căutare semantică în baza de date de cărți
- **OpenAI GPT Integration**: Conversație naturală și inteligentă
- **Function Calling**: Tool automat pentru rezumate detaliate
- **Filtru de Limbaj**: Protecție împotriva limbajului nepotrivit
- **Două Interfețe**: Streamlit (web) și CLI
- **12 Cărți Clasice**: Bază de date cu opere literare importante

## Tehnologii Folosite

### Limbaje de Programare
- **Python 3.8+** - Limbajul principal

### Dependințe Principale
- **openai** (1.3.0+) - API pentru GPT-4
- **chromadb** (0.4.0+) - Vector database pentru RAG
- **streamlit** (1.28.0+) - Interfață web
- **python-dotenv** (1.0.0+) - Management variabile de mediu

### Modele AI
- **GPT-4o-mini** - Model conversațional
- **text-embedding-3-small** - Model pentru embeddings

## Structura Proiectului

```
book-recommendation-chatbot/
│
├── app.py                  # Aplicația principală
├── book_summaries.txt      # Baza de date cu cărți
├── requirements.txt        # Dependințe Python
├── .env                    # Variabile de mediu (creat de utilizator)
├── .env.example            # Exemplu pentru .env
└── README.md              # Documentație
```

## Instalare

### 1. Clonează repository-ul
```bash
git clone <repository-url>
cd book-recommendation-chatbot
```

### 2. Creează un environment virtual (recomandat)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalează dependințele
```bash
pip install -r requirements.txt
```

### 4. Configurează OpenAI API Key

Creează un fișier `.env` în directorul principal:
```bash
cp .env.example .env
```

Editează `.env` și adaugă cheia ta API:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Pentru a obține o cheie API:
1. Mergi la [OpenAI Platform](https://platform.openai.com/)
2. Creează un cont sau autentifică-te
3. Navighează la API Keys
4. Creează o cheie nouă

## Rulare

### Interfață Web (Streamlit) - Recomandat
```bash
streamlit run app.py
```
Aplicația va deschide automat browserul la `http://localhost:8501`

### Interfață CLI (Command Line)
```bash
python app.py --cli
```

## Utilizare

### Exemple de întrebări valide:
- "Vreau o carte despre prietenie și magie"
- "Ce recomanzi pentru cineva care iubește poveștile de război?"
- "Aș vrea ceva despre aventură și curaj"
- "Caut o carte distopică despre societate"
- "Îmi place fantasy-ul epic"
- "Vreau să citesc despre dragoste și prejudecăți sociale"
- "Mă interesează cărți cu dezvoltare personală"

### Flux de funcționare:
1. Utilizatorul pune o întrebare despre preferințele de lectură
2. Sistemul verifică limbajul (filtru de profanitate)
3. ChromaDB caută semantic cele mai relevante cărți
4. GPT-4 analizează rezultatele și alege cea mai potrivită carte
5. Function calling obține automat rezumatul detaliat
6. Utilizatorul primește recomandarea completă cu explicații

## Componente Tehnice

### 1. **ProfanityFilter**
- Detectează limbaj nepotrivit în română și engleză
- Previne trimiterea către API-ul OpenAI
- Răspunde politicos cu cerere de reformulare

### 2. **BookRAG**
- Gestionează ChromaDB collection
- Creează embeddings cu OpenAI
- Implementează căutare semantică

### 3. **BookRecommendationChatbot**
- Orchestrează întregul proces
- Integrează RAG cu GPT-4
- Gestionează function calling

### 4. **get_summary_by_title Tool**
- Funcție înregistrată ca OpenAI tool
- Returnează rezumate detaliate (10-15 rânduri)
- Se apelează automat după recomandare

## Baza de Date

Aplicația include 12 cărți clasice:
- 1984 (George Orwell)
- The Hobbit (J.R.R. Tolkien)
- To Kill a Mockingbird (Harper Lee)
- Pride and Prejudice (Jane Austen)
- The Lord of the Rings (J.R.R. Tolkien)
- Harry Potter and the Sorcerer's Stone (J.K. Rowling)
- The Great Gatsby (F. Scott Fitzgerald)
- War and Peace (Leo Tolstoy)
- The Alchemist (Paulo Coelho)
- Dune (Frank Herbert)
- The Catcher in the Rye (J.D. Salinger)
- One Hundred Years of Solitude (Gabriel García Márquez)

## Troubleshooting

### Eroare: "OPENAI_API_KEY not found"
- Asigură-te că ai creat fișierul `.env`
- Verifică că ai adăugat cheia API corectă

### Eroare: "ChromaDB collection already exists"
- Este normal la rulări ulterioare
- Aplicația va folosi colecția existentă

### Streamlit nu pornește
- Verifică că portul 8501 este liber
- Încearcă: `streamlit run app.py --server.port 8502`

### Rate limiting OpenAI
- Folosește GPT-3.5-turbo în loc de GPT-4 (mai ieftin)
- Modifică în cod: `CHAT_MODEL = "gpt-3.5-turbo"`

## Costuri Estimate

Pentru 100 de conversații tipice:
- GPT-4o-mini: ~$0.50 - $1.00
- Embeddings: ~$0.01
- Total: Sub $1.50

## Dezvoltare Viitoare

Posibile îmbunătățiri:
- [ ] Adăugare mai multe cărți în baza de date
- [ ] Salvare istoric conversații
- [ ] Export recomandări în PDF
- [ ] Sistem de rating pentru recomandări
- [ ] Suport multilingv complet
- [ ] Integrare cu API-uri de librării

## Licență

MIT License - Vezi LICENSE pentru detalii

## Contact

Pentru întrebări sau sugestii, deschide un issue în repository.

## Acknowledgments

- OpenAI pentru GPT și embedding models
- ChromaDB pentru vector database
- Streamlit pentru framework-ul UI