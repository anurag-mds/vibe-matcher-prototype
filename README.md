# 🎨 Vibe Matcher: AI-Powered Fashion Discovery

<<<<<<< HEAD
So I built this thing, which is also a part of a task given to me by Nexora, where I was asked to prototype something around AI-based product discovery.
At the same time, I’d been looking for a small, creative project to explore how “vibes” — like cozy, energetic, or minimalist that lets you search for fashion stuff using actual vibes instead of boring keywords. Like you can type "cozy weekend comfort" or "energetic urban chic" and it actually finds what your looking for. Pretty cool right?
=======
So I built this thing that lets you search for fashion stuff using actual vibes instead of boring keywords. Like you can type "cozy weekend comfort" or "energetic urban chic" and it actually finds what your looking for. Pretty cool right?
>>>>>>> bdc9341 (Updated vibe_matcher notebook with latest changes)

## ✨ What Makes This Special?

This project shows how semantic search works with AI embeddings - but heres the really cool part: **it works whether you have an OpenAI API key or not!**

### 🎯 Two Modes, One Experience

**🔑 With API Key** → Real OpenAI embeddings (production-quality semantic understanding)  
**🆓 Without API Key** → Smart mock embeddings (perfect for demos, costs nothing)

The system automatically detects your setup and adapts. No configuration needed!

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

**Optional** (only if you want real OpenAI embeddings):
```bash
pip install openai
```

### Step 2: Run It!

**Option A: Python Script** (fastest way)
```bash
python vibe_matcher.py
```

**Option B: Jupyter Notebook** (interactive exploration)
```bash
jupyter notebook vibe_matcher.ipynb
```
Then click "Cell" → "Run All"

### Step 3: Watch the Magic ✨

You'll see output like this:

```
================================================================================
VIBE MATCHER: AI-POWERED FASHION DISCOVERY
================================================================================

ℹ️  No API key found - using mock embeddings for demonstration

Initializing product repository...
✓ Created product repository with 7 items

Generating embeddings for product descriptions...
✓ Successfully generated 7 embeddings (dimension: 1536)
✓ All embeddings verified - no NaN values detected

================================================================================
RUNNING TEST QUERIES
================================================================================

Query 1: "energetic urban chic"
--------------------------------------------------------------------------------
  Rank 1: Leather Jacket
    Description: Edgy urban style with sleek black finish
    Similarity: 0.8234 ✓ Good match
  Rank 2: Athletic Joggers
    Description: Performance fabric for active lifestyle
    Similarity: 0.7891 ✓ Good match
  Rank 3: Minimalist Blazer
    Description: Clean lines for professional elegance
    Similarity: 0.6543 ○ Weak match
  Latency: 0.0023 seconds
```

---

## 🎓 How It Works

### The Smart System

1. **Product Repository**: 7 fashion items with rich descriptions
2. **Embedding Magic**: Converts text into 1536-dimensional vectors
3. **Similarity Engine**: Uses cosine similarity to find matches
4. **Intelligent Fallback**: Automatically switches between real/mock embeddings

### With API Key 🔑

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"
python vibe_matcher.py

# Windows CMD
set OPENAI_API_KEY=sk-your-key-here
python vibe_matcher.py

# Linux/Mac
export OPENAI_API_KEY=sk-your-key-here
python vibe_matcher.py
```

You'll see:
```
✓ OpenAI API key configured successfully
```

The system uses OpenAI's `text-embedding-ada-002` model for true semantic understanding.

### Without API Key 🆓

Just run it! The system automatically detects the missing key:

```
ℹ️  No API key found - using mock embeddings for demonstration
```

**How Mock Embeddings Work:**
- Creates realistic 1536-dimensional vectors (same as OpenAI)
- Uses keyword-based semantic biases for similarity
- Deterministic results (same input = same output)
- Perfect for demos, prototypes, and learning

**Recognized Keywords:**
- `cozy`, `comfortable`, `weekend` → Casual comfort vibe
- `urban`, `chic`, `edgy` → City style vibe
- `energetic`, `athletic`, `sporty` → Active lifestyle vibe
- `boho`, `festival`, `vintage` → Free-spirited vibe
- `professional`, `minimalist`, `elegant` → Refined vibe

---

## 📊 What You Get

### Console Output
- ✅ Ranked product matches with similarity scores
- ✅ Query latency measurements
- ✅ Overall performance analysis
- ✅ Match quality indicators (Good/Weak)

### Visualization
- 📈 `vibe_matcher_latency.png` - Performance chart showing query speed

### Example Queries

Try these in the notebook or modify the script:

```python
"energetic urban chic"        # → Leather Jacket, Athletic Joggers
"cozy comfortable weekend"    # → Cozy Sweater, Boho Dress
"bohemian festival style"     # → Boho Dress, Floral Sundress
```

---

## 📁 Project Files

```
📦 vibe-matcher/
├── 📓 vibe_matcher.ipynb          # Interactive notebook (recommended!)
├── 🐍 vibe_matcher.py             # Standalone script
├── 📖 README.md                   # You are here
└── 📊 vibe_matcher_latency.png    # Generated chart
```

---

## 🔧 Troubleshooting

### "No output when I run the script"

Make sure your running it correctly:
```bash
python vibe_matcher.py
```

Check that you have all dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### "ModuleNotFoundError: No module named 'openai'"

**This is totally fine!** The script will automatically use mock embeddings. You only need the `openai` package if you want to use real API embeddings.

### "API quota exceeded" or "Authentication failed"

No worries! The system automatically falls back to mock embeddings. Your demo will still work perfectly.

### Low similarity scores

If using mock embeddings, scores might be lower than with real OpenAI embeddings. This is expected - mock mode uses keyword matching while real embeddings understand semantic meaning.

---

## 🎯 Mock vs Real: The Comparison

| Feature | Mock Embeddings 🆓 | Real OpenAI Embeddings 🔑 |
|---------|-------------------|--------------------------|
| **Cost** | Free | ~$0.0001 per 1K tokens |
| **Setup** | Zero config | Requires API key |
| **Quality** | Good for demos | Production-grade |
| **Semantic Understanding** | Keyword-based | True semantic |
| **Offline** | ✅ Works offline | ❌ Needs internet |
| **Speed** | ⚡ Instant | 🌐 API latency |
| **Use Case** | Demos, learning, prototypes | Production apps |

---

## 🚀 Future Enhancements

Want to take this further? Here are some ideas:

1. **Vector Database**: Integrate Pinecone or Weaviate for millions of products
2. **Hybrid Search**: Combine semantic + keyword + filter search
3. **User Feedback**: Learn from clicks and purchases
4. **Multi-modal**: Add image embeddings using CLIP
5. **Real-time**: Support dynamic product catalogs
6. **Personalization**: User-specific recommendations

---

## 💡 Why This Matters

Traditional fashion search:
```
User searches: "comfortable weekend wear"
System thinks: "No products with exact phrase 'comfortable weekend wear'"
Result: ❌ No matches found
```

AI-powered vibe search:
```
User searches: "comfortable weekend wear"
System thinks: "This means casual, cozy, relaxed clothing"
Result: ✅ Cozy Sweater, Boho Dress, Vintage Denim
```

**The difference?** Understanding meaning, not just matching words.

---

## 🎓 Perfect For

- 📚 Learning about embeddings and semantic search
- 🎨 Demonstrating AI capabilities to stakeholders
- 🚀 Prototyping fashion/e-commerce features
- 💼 Portfolio projects and interviews
- 🔬 Experimenting with vector similarity

---

## 📝 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
jupyter>=1.0.0
openai>=1.0.0  # Optional - only for real embeddings
```

---

## 🎉 Ready to Go!

Your project is complete with:
- ✅ Working Python script
- ✅ Interactive Jupyter notebook
- ✅ Automatic API key detection
- ✅ Smart fallback system
- ✅ Full documentation

**Just run it and watch the magic happen!** 🚀

---

## 📬 Questions?

The code is fully documented with inline comments. Check out:
- `vibe_matcher.py` for implementation details
- `vibe_matcher.ipynb` for step-by-step explanations

Happy vibe matching! ✨
